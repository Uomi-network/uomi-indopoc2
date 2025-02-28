import sys
import time
import os
import redis
import base64
import threading
import json
import traceback
import hashlib
import math
import collections

import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurations
############################################

NODE_ID = os.getenv('NODE_ID', 1)
PROMPTS_FILE_PATH = './prompts_deepseek.txt'

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASS = os.getenv('REDIS_PASS', '')
REDIS_PORT = 6379
REDIS_NODES_DB = 0
REDIS_ASSIGNMENTS_DB = 12
REDIS_CHECKS_DB = 13
REDIS_COMPLETITION_DB = 14
REDIS_PROMPTS_DB = 15
REDIS_NODE_INFERENCES_DB = NODE_ID

SEED = 42
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K_EXECUTION = 5
TOK_K_CHECK = 10
BATCH_SIZE_CHECK = 5
BATCH_SIZE_INFERENCE = 5

NODES = [
  1, # RTX 4090
  2, # RTX A6000
  3, # H100 SXM
  4, # L40S
  5, # A100 SXM
]

# Redis Connections
############################################

r_nodes_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_NODES_DB, password=REDIS_PASS)
r_assignments_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_ASSIGNMENTS_DB, password=REDIS_PASS)
r_checks_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_CHECKS_DB, password=REDIS_PASS)
r_completition_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_COMPLETITION_DB, password=REDIS_PASS)
r_prompts_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_PROMPTS_DB, password=REDIS_PASS)
r_node_inferences_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_NODE_INFERENCES_DB, password=REDIS_PASS)

# Setup model
############################################

# - Set reproducibility settings
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# - Some additional flags to help reproducibility in certain cases:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# - Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# - Set the model and tokenizer
model_name = "casperhansen/deepseek-r1-distill-qwen-14b-awq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True)
model.to(device)
model.eval()  # put model in eval mode (no dropout, etc.)

# Functions
############################################

# This function generate a unique hash for a given string
def hash_string(input_string):
  # input_bytes = input_string.encode('utf-8')
  # return base64.b64encode(input_bytes).decode('utf-8')
  length = 64
  hash_object = hashlib.sha256(input_string.encode())
  return hash_object.hexdigest()[:length]

def execute_batch_inferences(batch_prompts, batch_keys):
  """
  Execute and store inferences for a batch of prompts using true batch processing.
  
  Args:
      batch_prompts: A list of prompts to process in batch
      batch_keys: A list of keys corresponding to each prompt
      
  Returns:
      A list of inference results matching each prompt in the batch
  """
  time_start = time.time()
  batch_size = len(batch_prompts)
  
  # Tokenize all prompts
  all_input_ids = []
  for prompt in batch_prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    all_input_ids.append(ids)
  
  # Initialize tracking variables for each sequence in the batch
  all_execution_data = [[] for _ in range(batch_size)]
  all_output_tokens = [[] for _ in range(batch_size)]
  first_new_token_ids = [None for _ in range(batch_size)]
  active_batch_indices = list(range(batch_size))
  
  # Create tracking for prompt lengths
  prompt_lengths = [ids.shape[1] for ids in all_input_ids]
  max_prompt_length = max(prompt_lengths)
  
  # Pre-allocate tensors with enough space for full sequence (prompt + all output tokens)
  # Add extra padding to avoid index errors
  full_sequence_length = max_prompt_length + MAX_NEW_TOKENS + 10  # Added safety margin
  
  # Create padded input tensors with attention masks
  batched_input_ids = torch.zeros((batch_size, full_sequence_length), dtype=torch.long, device=device)
  batched_attention_masks = torch.zeros((batch_size, full_sequence_length), dtype=torch.long, device=device)
  
  # Fill in the prompt portions
  for i, input_ids in enumerate(all_input_ids):
    seq_len = input_ids.shape[1]
    batched_input_ids[i, :seq_len] = input_ids.squeeze(0)
    batched_attention_masks[i, :seq_len] = 1
  
  # Track current token position for each sequence (starts at end of prompt)
  current_positions = prompt_lengths.copy()
  
  # Process tokens step by step
  for step in range(MAX_NEW_TOKENS):
    print(f"Step execute_batch_inference {step + 1}/{MAX_NEW_TOKENS}, active prompts: {len(active_batch_indices)}/{batch_size}")
    
    if not active_batch_indices:
      break  # All inferences have completed
    
    # Get active batch
    active_input_ids = batched_input_ids[active_batch_indices]
    active_attention_masks = batched_attention_masks[active_batch_indices]
    
    # Forward pass on the active batch
    ts = time.time()
    outputs = model(input_ids=active_input_ids, attention_mask=active_attention_masks)
    print(f"- time for forward pass (batch of {len(active_batch_indices)}): {time.time() - ts}")
    
    # Process each active inference
    to_remove = []
    for batch_pos, original_idx in enumerate(active_batch_indices):
      # Get the position of the last token in this sequence
      last_pos = current_positions[original_idx] - 1
      next_token_logits = outputs.logits[batch_pos, last_pos, :].unsqueeze(0)
      
      # Apply temperature (if not 1.0)
      if TEMPERATURE != 1.0:
        next_token_logits = next_token_logits / TEMPERATURE
      
      # Optional top-p filtering
      if TOP_P < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
        sorted_logits_1d = sorted_logits[0]
        sorted_indices_1d = sorted_indices[0]
        
        sorted_probs = F.softmax(sorted_logits_1d, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        cutoff_idx = torch.searchsorted(cumulative_probs, TOP_P)
        cutoff_idx = cutoff_idx.clamp(max=sorted_probs.size(-1) - 1)
        sorted_logits_1d[cutoff_idx + 1:] = float('-inf')
        
        # Scatter back
        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
        next_token_logits[0].scatter_(0, sorted_indices_1d, sorted_logits_1d)
      
      # Convert to probabilities
      probs = F.softmax(next_token_logits, dim=-1)
      
      # Get top-k tokens by probability
      top_probs, top_indices = probs.topk(TOP_K_EXECUTION, dim=-1)
      execution_data_top_k = []
      for idx in top_indices[0]:
        token_str = tokenizer.decode([idx.item()])
        prob = probs[0, idx].item()
        execution_data_top_k.append({
          "str": token_str,
          "prob": prob,
          "id": idx.item()
        })
      
      # Sample from the top-k tokens
      next_token_id = top_indices.select(-1, torch.multinomial(top_probs, num_samples=1).item()).unsqueeze(0)
      selected_token_id = next_token_id.item()
      selected_token_str = tokenizer.decode([selected_token_id])
      selected_token_prob = probs[0, selected_token_id].item()
      
      # Record the token data
      all_execution_data[original_idx].append({
        "str": selected_token_str,
        "prob": selected_token_prob,
        "id": selected_token_id,
        "top_k": execution_data_top_k
      })
      all_output_tokens[original_idx].append(selected_token_id)
      
      # Record the first token if this is the first step
      if step == 0:
        first_new_token_ids[original_idx] = selected_token_id
      
      # Add the selected token to the input sequence for next iteration
      pos = current_positions[original_idx]
      batched_input_ids[original_idx, pos] = selected_token_id
      batched_attention_masks[original_idx, pos] = 1
      current_positions[original_idx] += 1
      
      # Check if this sequence has completed (e.g., EOS token)
      # For now, we're just using the max length
    
  # Build final outputs
  results = []
  for i, key in enumerate(batch_keys):
    # Get the full output including the prompt
    output_tokens = all_output_tokens[i]
    full_sequence = batched_input_ids[i, :current_positions[i]].tolist()
    output = tokenizer.decode(full_sequence, skip_special_tokens=False)
    
    result = {
      "key": key,
      "output": output,
      "output_tokens": output_tokens,
      "execution_data": all_execution_data[i],
      "executed_by": NODE_ID,
      "executed_in": time.time() - time_start,
      "full_sequence_length": full_sequence_length
    }
    results.append(json.dumps(result))
    print(f"✅ Inference {i} completed")
  
  print(f"✅ Batch inference completed: {time.time() - time_start}s for {batch_size} prompts")
  return results

def execute_batch_checks(batch_checks):
  """
  Execute and store checks for a batch of inferences using true batch processing.
  
  Args:
      batch_checks: A list of inference data to check in batch
      
  Returns:
      A list of check results matching each inference in the batch
  """
  time_start = time.time()
  
  # Parse all inferences from the batch
  inferences = [json.loads(inference) for inference in batch_checks]
  batch_size = len(inferences)
  
  # Get prompts for all inferences
  prompts = [r_prompts_db.get(inference["key"]).decode('utf-8') for inference in inferences]
  
  # Calculate maximum sequence length we'll need (prompt + max output tokens)
  max_output_tokens = max(len(inference["output_tokens"]) for inference in inferences)
  max_output_tokens = min(max_output_tokens, MAX_NEW_TOKENS)
  
  # Tokenize all prompts
  all_input_ids = []
  for prompt in prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    all_input_ids.append(ids)
  
  # Initialize tracking variables
  all_check_data = [[] for _ in range(batch_size)]
  all_check_results = [True for _ in range(batch_size)]
  active_batch_indices = list(range(batch_size))
  
  # Create tracking for prompt lengths
  prompt_lengths = [ids.shape[1] for ids in all_input_ids]
  max_prompt_length = max(prompt_lengths)
  
  # Set full_sequence_length as the maximum of inferences full_sequence_length
  full_sequence_length = max(inference["full_sequence_length"] for inference in inferences)
  
  # Create padded input tensors with attention masks
  batched_input_ids = torch.zeros((batch_size, full_sequence_length), dtype=torch.long, device=device)
  batched_attention_masks = torch.zeros((batch_size, full_sequence_length), dtype=torch.long, device=device)
  
  # Fill in the prompt portions
  for i, input_ids in enumerate(all_input_ids):
    seq_len = input_ids.shape[1]
    batched_input_ids[i, :seq_len] = input_ids.squeeze(0)
    batched_attention_masks[i, :seq_len] = 1
  
  # Track current token position for each sequence (starts at end of prompt)
  current_positions = prompt_lengths.copy()
  
  # Process tokens step by step
  for step in range(max_output_tokens):
    print(f"Step execute_batch_check {step + 1}/{max_output_tokens}, active inferences: {len(active_batch_indices)}/{batch_size}")
    
    if not active_batch_indices:
      break  # All inferences have completed or failed checks
    
    # Get active batch
    active_input_ids = batched_input_ids[active_batch_indices]
    active_attention_masks = batched_attention_masks[active_batch_indices]
    
    # Forward pass on the active batch
    ts = time.time()
    outputs = model(input_ids=active_input_ids, attention_mask=active_attention_masks)
    print(f"- time for forward pass (batch of {len(active_batch_indices)}): {time.time() - ts}")
    
    # Get logits for the next token prediction (last token in each sequence)
    next_token_logits = []
    for i, batch_idx in enumerate(active_batch_indices):
      # Get the position of the last token in this sequence
      last_pos = current_positions[batch_idx] - 1
      next_token_logits.append(outputs.logits[i, last_pos, :].unsqueeze(0))
    
    next_token_logits = torch.cat(next_token_logits, dim=0)
    
    # Apply temperature (if not 1.0)
    ts = time.time()
    if TEMPERATURE != 1.0:
      next_token_logits = next_token_logits / TEMPERATURE
    print(f"- time for temperature: {time.time() - ts}")
    
    # Optional top-p filtering
    if TOP_P < 1.0:
      ts = time.time()
      for i in range(len(active_batch_indices)):
        logits = next_token_logits[i].unsqueeze(0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_logits_1d = sorted_logits[0]
        sorted_indices_1d = sorted_indices[0]
        
        sorted_probs = F.softmax(sorted_logits_1d, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        cutoff_idx = torch.searchsorted(cumulative_probs, TOP_P)
        cutoff_idx = cutoff_idx.clamp(max=sorted_probs.size(-1) - 1)
        sorted_logits_1d[cutoff_idx + 1:] = float('-inf')
        
        # Scatter back
        next_token_logits[i] = torch.full_like(next_token_logits[i], float('-inf'))
        next_token_logits[i].scatter_(0, sorted_indices_1d, sorted_logits_1d)
      print(f"- time for top-p: {time.time() - ts}")
    
    # Convert to probabilities
    probs = F.softmax(next_token_logits, dim=-1)
    
    # Process each active inference
    to_remove = []
    for batch_pos, original_idx in enumerate(active_batch_indices):
      inference = inferences[original_idx]
      
      # Skip if we're past the tokens for this inference
      if step >= len(inference["output_tokens"]):
        to_remove.append(batch_pos)
        continue
      
      # Get target token for this inference at this step
      current_token_id = inference["output_tokens"][step]
      current_token_str = tokenizer.decode([current_token_id])
      
      # Check if target token is in top-k
      ts = time.time()
      top_probs, top_indices = probs[batch_pos].topk(TOK_K_CHECK + 5)
      print(f"- time for top_k (inference {original_idx}): {time.time() - ts}")
      
      # Build top-k data and check if current token is in it
      ts = time.time()
      check_data_top_k = []
      current_token_prob = None
      
      for i, (idx, prob_val) in enumerate(zip(top_indices, top_probs)):
        token_str = tokenizer.decode([idx.item()])
        prob = prob_val.item()
        check_data_top_k.append({
          "str": token_str,
          "prob": prob,
          "id": idx.item()
        })
        if idx.item() == current_token_id and i < TOK_K_CHECK:
          current_token_prob = float(prob)
      
      print(f"- time for check_data_top_k (inference {original_idx}): {time.time() - ts}")
      
      # Record check results
      token_data = {
        "str": current_token_str,
        "prob": current_token_prob,
        "id": current_token_id,
        "top_k": check_data_top_k
      }
      all_check_data[original_idx].append(token_data)
      
      # Check if token is not in top-k
      if current_token_prob is None:
        all_check_results[original_idx] = False
        print(f"❌ Inference {original_idx} - Current token: '{current_token_str}' -> not found in top-{TOK_K_CHECK}")
        to_remove.append(batch_pos)
        continue
      
      # Add the predicted token to the input sequence for next iteration
      ts = time.time()
      pos = current_positions[original_idx]
      batched_input_ids[original_idx, pos] = current_token_id
      batched_attention_masks[original_idx, pos] = 1
      current_positions[original_idx] += 1
      print(f"- time for update input_ids (inference {original_idx}): {time.time() - ts}")
    
    # Remove completed or failed inferences
    for idx in sorted(to_remove, reverse=True):
      del active_batch_indices[idx]
  
  # Build final results
  total_time = time.time() - time_start
  results = []
  
  for i, inference in enumerate(inferences):
    result = {
      "key": inference["key"],
      "check_result": all_check_results[i],
      "check_data": all_check_data[i],
      "checked_by": NODE_ID,
      "checked_in": total_time,
      "executed_by": inference["executed_by"],
      "executed_in": inference["executed_in"]
    }
    results.append(json.dumps(result))
    print(f"✅ Check {i} completed")
  
  print(f"✅ Batch check completed: {total_time}s for {batch_size} inferences")
  return results

def run():
  print("🧠 Node " + str(NODE_ID) + " is looping run...")
  
  try:
    remaining = 0

    # Start execution of the inferences (from r_assignments_db) and store the result in the node's db
    # NOTE: Ignore execution if it is already stored in the node's db
    r_assignments_db_keys = r_assignments_db.keys()
    r_assignments_db_keys = [key.decode('utf-8') for key in r_assignments_db_keys]
    r_assignments_db_keys_of_node = [key for key in r_assignments_db_keys if key.split("_")[0] == str(NODE_ID)]
    r_assignments_db_keys_of_node = [key.split("_")[1] for key in r_assignments_db_keys_of_node]
    r_node_inferences_db_keys = r_node_inferences_db.keys()
    r_node_inferences_db_keys = [key.decode('utf-8') for key in r_node_inferences_db_keys]
    
    # Collect prompts for batch processing
    batch_prompts = []
    batch_keys = []

    for key in r_assignments_db_keys_of_node:
      if r_node_inferences_db_keys.__contains__(key):
        print("Skipping inference: " + str(key))
      elif len(batch_prompts) < BATCH_SIZE_INFERENCE:
        print("Taking inference: " + str(key))
        prompt = r_prompts_db.get(key).decode('utf-8')
        batch_prompts.append(prompt)
        batch_keys.append(key)
      else:
        print("Remaining op inference: " + str(remaining))
        remaining += 1
    
    # Execute batch inference if there are prompts to process
    if len(batch_prompts) > 0:
      print(f"Executing batch inference for {len(batch_prompts)} prompts")
      results = execute_batch_inferences(batch_prompts, batch_keys)
      # Store results in the node's db
      for key, result in zip(batch_keys, results):
        r_node_inferences_db.set(key, result)

    # Update ping
    r_nodes_db.set(str(NODE_ID), 1, ex=600)

    # Take list of other nodes from the r_nodes_db
    nodes = [node for node in NODES if node != int(NODE_ID)]
    # Sort nodes randomly
    random.shuffle(nodes)

    # Loop through the nodes, for each node take its inferences and execute the check
    batch_checks = []
    for node in nodes:
      print("Checking node: " + str(node))
      # Try to connect to the node's db, if db not exists, skip the node
      try:
        node_inferences_db = r_node_inferences_db if node == NODE_ID else redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=node, password=REDIS_PASS)
      except:
        print("Skipping node: " + str(node))
        continue
      node_inferences_db_keys = node_inferences_db.keys()
      r_checks_db_keys = r_checks_db.keys()
      r_checks_db_keys = [key.decode('utf-8') for key in r_checks_db_keys]
      for key in node_inferences_db_keys:
        check_key = str(NODE_ID) + "_" + str(node) + "_" + key.decode('utf-8')
        if r_checks_db_keys.__contains__(check_key):
          print("Skipping check: " + str(check_key))
        elif len(batch_checks) < BATCH_SIZE_CHECK:
          print("Taking check: " + str(check_key))
          inference = node_inferences_db.get(key).decode('utf-8')
          batch_checks.append((check_key, inference))
        else:
          print("Remaining op check: " + str(remaining))
          remaining += 1
    if len(batch_checks) > 0:
      print("Executing checks: " + str(len(batch_checks)))
      check_keys = [check[0] for check in batch_checks]
      inferences = [check[1] for check in batch_checks]
      results = execute_batch_checks(inferences)
      for key, result in zip(check_keys, results):
        r_checks_db.set(key, result)

    # Store the node's db in the completition db
    r_completition_db.set(str(NODE_ID), remaining)

    # Update ping
    r_nodes_db.set(str(NODE_ID), 1, ex=600)

    print("✅ Node " + str(NODE_ID) + " completed the run loop.")
  except Exception as e:
    print("❌ Node " + str(NODE_ID) + " failed to complete the run loop.")
    print(traceback.format_exc())
    r_completition_db.set(str(NODE_ID), -1)

def setup():
  # Read the prompts from the file (one inference per line)
  with open(PROMPTS_FILE_PATH, 'r') as f:
    prompts = f.readlines()
  # Normalize prompts by remove last character if is a new line
  prompts = [inference.rstrip('\n') for inference in prompts]
  # Store every inference in the redis db using the hash_string of the inference as key (if not already stored)
  r_prompts_db_keys = r_prompts_db.keys()
  r_prompts_db_keys = [key.decode('utf-8') for key in r_prompts_db_keys]
  for prompt in prompts:
    prompt_hash = hash_string(prompt)
    if not r_prompts_db_keys.__contains__(prompt_hash):
      r_prompts_db.set(prompt_hash, prompt)
      print("Store prompt: " + str(prompt_hash))
    else:
      print("Skipping store prompt: " + str(prompt_hash))
  # Assign each prompt in the r_prompts_db to the node, being sure that the prompt is not already assigned to another node
  # and every node should have the same number of prompts.
  r_prompts_db_keys = r_prompts_db.keys()
  r_prompts_db_keys = [key.decode('utf-8') for key in r_prompts_db_keys]
  r_prompts_db_keys = random.sample(r_prompts_db_keys, len(r_prompts_db_keys))
  r_assignments_db_keys = r_assignments_db.keys()
  r_assignments_db_keys = [key.decode('utf-8') for key in r_assignments_db_keys]
  r_assignments_db_keys_prompts = [key.split("_")[1] for key in r_assignments_db_keys]
  r_assignments_db_keys_nodes = [key.split("_")[0] for key in r_assignments_db_keys]
  assignments_per_node = math.floor(len(r_prompts_db_keys) / len(NODES))
  assignments_already_assigned_to_node = r_assignments_db_keys_nodes.count(str(NODE_ID))
  for r_prompts_db_key in r_prompts_db_keys:
    if r_assignments_db_keys_prompts.__contains__(r_prompts_db_key):
      print("Skipping assignment already assigned: " + str(r_prompts_db_key))
    elif assignments_already_assigned_to_node >= assignments_per_node:
      print("Skipping assignment limit reached: " + str(r_prompts_db_key))
    else:
      assignment_key = str(NODE_ID) + "_" + r_prompts_db_key
      r_assignments_db.set(assignment_key, r_prompts_db_key)
      print("Store assignment: " + str(assignment_key))
      assignments_already_assigned_to_node += 1

# Main
############################################

if __name__ == '__main__':
  print("🚀 Node " + str(NODE_ID) + " is running...")

  # Setup
  setup()

  # Run
  while True:
    run()
    time.sleep(1)

