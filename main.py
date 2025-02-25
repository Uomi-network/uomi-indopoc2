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
PROMPTS_FILE_PATH = './prompts.txt'

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
BACK_SIZE_CHECK = 2

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
model_name = "casperhansen/mistral-small-24b-instruct-2501-awq"
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

# This function execute the inference and return the result
def execute_inference(prompt, key):
  time_start = time.time()
  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

  execution_data = []
  first_new_token_id = None

  for step in range(MAX_NEW_TOKENS):
    print(f"Step execute_inference {step + 1}/{MAX_NEW_TOKENS}")
    # Forward pass to get raw logits
    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]

    # Apply temperature (if not 1.0)
    if TEMPERATURE != 1.0:
      next_token_logits = next_token_logits / TEMPERATURE
    
    # Optional top-p filtering (here, top_p=1.0 => no filtering)
    if TOP_P < 1.0:
      sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
      sorted_logits_1d = sorted_logits[0]
      sorted_indices_1d = sorted_indices[0]

      sorted_probs = F.softmax(sorted_logits_1d, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

      cutoff_idx = torch.searchsorted(cumulative_probs, TOP_P)
      cutoff_idx = cutoff_idx.clamp(max=sorted_probs.size- 1)
      sorted_logits_1d[cutoff_idx + 1:] = float('-inf')

      # Scatter back
      next_token_logits = torch.full_like(next_token_logits, float('-inf'))
      next_token_logits[0].scatter_(0, sorted_indices_1d, sorted_logits_1d)

    # Convert to probabilities
    probs = F.softmax(next_token_logits, dim=-1)  # shape: [1, vocab_size]

    # Print the top-k tokens by probability
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

    # GREEDY selection instead of sampling
    # This ensures full determinism.
    next_token_id = top_indices.select(-1, torch.multinomial(top_probs, num_samples=1).item()).unsqueeze(0)
    selected_token_id = next_token_id.item()
    selected_token_str = tokenizer.decode([selected_token_id])
    selected_token_prob = probs[0, selected_token_id].item()

    # Append the chosen token
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Append the execution data
    execution_data.append({
      "str": selected_token_str,
      "prob": selected_token_prob,
      "id": selected_token_id,
      "top_k": execution_data_top_k
    })
    first_new_token_id = selected_token_id if first_new_token_id is None else first_new_token_id

  output = tokenizer.decode(input_ids[0], skip_special_tokens=False)
  # output_tokens_all = input_ids[0].tolist()
  # # Remove ids from output_tokens_all before the first new token
  # output_tokens = []
  # first_new_token_found = False
  # for token_id in output_tokens_all:
  #   if first_new_token_found:
  #     output_tokens.append(token_id)
  #   if first_new_token_found == False and token_id == first_new_token_id:
  #     first_new_token_found = True
  #     output_tokens.append(token_id)
  output_tokens = []
  for execution_data_step in execution_data:
    output_tokens.append(execution_data_step["id"])

  result = {
    "key": key,
    "output": output,
    "output_tokens": output_tokens,
    "execution_data": execution_data,
    "executed_by": NODE_ID,
    "executed_in": time.time() - time_start
  }
  return json.dumps(result)

def execute_batch_checks(batch_checks):
  """
  Execute and store checks for a batch of inferences.
  
  Args:
      batch_checks: A list of inference data to check in batch
      
  Returns:
      A list of check results matching each inference in the batch
  """
  time_start = time.time()
  
  # Parse all inferences from the batch
  inferences = [json.loads(inference) for inference in batch_checks]
  
  # Get prompts for all inferences
  prompts = [r_prompts_db.get(inference["key"]).decode('utf-8') for inference in inferences]
  
  # Prepare batch data structures
  batch_size = len(inferences)
  all_input_ids = [tokenizer.encode(prompt, return_tensors="pt").to(device) for prompt in prompts]
  all_check_data = [[] for _ in range(batch_size)]
  all_check_results = [True for _ in range(batch_size)]
  active_indices = list(range(batch_size))  # Indices of inferences still being processed
  
  # Iterate through token positions
  for step in range(MAX_NEW_TOKENS):
    print(f"Step execute_batch_check {step + 1}/{MAX_NEW_TOKENS}, active inferences: {len(active_indices)}/{batch_size}")
    
    if not active_indices:
      break  # All inferences have completed or failed checks
    
    # Process each active inference
    for batch_idx in active_indices[:]:  # Create a copy to allow removal during iteration
      input_ids = all_input_ids[batch_idx]
      inference = inferences[batch_idx]
      
      # Check if we've reached the end of this inference's tokens
      if step >= len(inference["output_tokens"]):
        active_indices.remove(batch_idx)
        continue
      
      # Forward pass to get raw logits
      ts = time.time()
      outputs = model(input_ids=input_ids)
      next_token_logits = outputs.logits[:, -1, :]
      print(f"- time for forward pass (inference {batch_idx}): {time.time() - ts}")
      
      # Apply temperature (if not 1.0)
      ts = time.time()
      if TEMPERATURE != 1.0:
        next_token_logits = next_token_logits / TEMPERATURE
      print(f"- time for temperature (inference {batch_idx}): {time.time() - ts}")
      
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
      probs = F.softmax(next_token_logits, dim=-1)  # shape: [1, vocab_size]
      
      # Check the current token's probability
      ts = time.time()
      check_data_top_k = []
      current_token_prob = None
      current_token_id = inference["output_tokens"][step]
      current_token_str = tokenizer.decode([current_token_id])
      top_probs, top_indices = probs.topk(TOK_K_CHECK + 5, dim=-1)
      print(f"- time for top_k (inference {batch_idx}): {time.time() - ts}")
      
      ts = time.time()
      index = 0
      for idx in top_indices[0]:
        index += 1
        token_str = tokenizer.decode([idx.item()])
        prob = probs[0, idx].item()
        check_data_top_k.append({
          "str": token_str,
          "prob": prob,
          "id": idx.item()
        })
        if idx == current_token_id and index <= TOK_K_CHECK:
          current_token_prob = float(prob)
      
      # Record the check data
      if current_token_prob is None:
        all_check_results[batch_idx] = False
        all_check_data[batch_idx].append({
          "str": current_token_str,
          "prob": current_token_prob,
          "id": current_token_id,
          "top_k": check_data_top_k
        })
        print(f"❌ Inference {batch_idx} - Current token: '{current_token_str}' -> not found in top-{TOK_K_CHECK}")
        active_indices.remove(batch_idx)  # Stop processing this inference
        continue
      
      print(f"- time for check_data_top_k (inference {batch_idx}): {time.time() - ts}")
      
      # Append the chosen token to the input_ids for next iteration
      ts = time.time()
      next_token_id = torch.tensor([[current_token_id]]).to(device)
      all_input_ids[batch_idx] = torch.cat([input_ids, next_token_id], dim=-1)
      print(f"- time for update input_ids (inference {batch_idx}): {time.time() - ts}")
      
      # Append the check result
      all_check_data[batch_idx].append({
        "str": current_token_str,
        "prob": current_token_prob,
        "id": current_token_id,
        "top_k": check_data_top_k
      })
  
  # Prepare the final results
  total_time = time.time() - time_start
  results = []
  
  for i, inference in enumerate(inferences):
    result = {
      "key": inference["key"],
      "check_result": all_check_results[i],
      "check_data": all_check_data[i],
      "checked_by": NODE_ID,
      "checked_in": total_time,  # Using the total batch time for simplicity
      "executed_by": inference["executed_by"],
      "executed_in": inference["executed_in"]
    }
    results.append(json.dumps(result))
    print(f"✅ Check {i} completed")
  
  print(f"✅ Batch check completed: {total_time}s for {batch_size} inferences")
  return results

completed = {}
for node in NODES:
  completed[node] = False
completed[NODE_ID] = False
def run():
  global completed
  print("🧠 Node " + str(NODE_ID) + " is looping run...")
  
  try:
    remaining = 0
    inference_to_run = None
    check_to_run = None

    # Start execution of the inferences (from r_assignments_db) and store the result in the node's db
    # NOTE: Ignore execution if it is already stored in the node's db
    prompts_runned_one = False
    if not completed[NODE_ID]:
      r_assignments_db_keys = r_assignments_db.keys()
      r_assignments_db_keys = [key.decode('utf-8') for key in r_assignments_db_keys]
      r_assignments_db_keys_of_node = [key for key in r_assignments_db_keys if key.split("_")[0] == str(NODE_ID)]
      r_assignments_db_keys_of_node = [key.split("_")[1] for key in r_assignments_db_keys_of_node]
      r_node_inferences_db_keys = r_node_inferences_db.keys()
      r_node_inferences_db_keys = [key.decode('utf-8') for key in r_node_inferences_db_keys]
      for key in r_assignments_db_keys_of_node:
        if r_node_inferences_db_keys.__contains__(key):
          print("Skipping inference: " + str(key))
        elif not prompts_runned_one:
          print("Executing inference: " + str(key))
          prompt = r_prompts_db.get(key).decode('utf-8')
          result = execute_inference(prompt, key)
          r_node_inferences_db.set(key, result)
          prompts_runned_one = True
        else:
          print("Remaining op inference: " + str(remaining))
          remaining += 1
          break
      if not prompts_runned_one:
        completed[NODE_ID] = True
        print("✅ Node " + str(NODE_ID) + " completed the inferences.")

    # Take list of other nodes from the r_nodes_db
    nodes = [node for node in NODES if node != int(NODE_ID)]
    # Sort nodes randomly
    random.shuffle(nodes)

    # Loop through the nodes, for each node take its inferences and execute the check
    batch_checks = []
    for node in nodes:
      if not completed[node]:
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
          elif len(batch_checks) < BACK_SIZE_CHECK:
            print("Taking check: " + str(check_key))
            inference = node_inferences_db.get(key).decode('utf-8')
            batch_checks.append((check_key, inference))
          else:
            print("Remaining op check: " + str(remaining))
            remaining += 1
            break
        if len(batch_checks) == 0:
          completed[node] = True
          print("✅ Node " + str(node) + " completed the checks.")
        if len(batch_checks) >= BACK_SIZE_CHECK:
          break
    if len(batch_checks) > 0:
      print("Executing checks: " + str(len(batch_checks)))
      check_keys = [check[0] for check in batch_checks]
      inferences = [check[1] for check in batch_checks]
      results = execute_batch_checks(inferences)
      print(results)

    # Store the node's db in the completition db
    r_completition_db.set(str(NODE_ID), remaining)

    # Update ping
    r_nodes_db.set(str(NODE_ID), 1, ex=300)

    print("✅ Node " + str(NODE_ID) + " completed the run loop.")
  except Exception as e:
    print("❌ Node " + str(NODE_ID) + " failed to complete the run loop.")
    print(traceback.format_exc())
    r_completition_db.set(str(NODE_ID), -1)


# Setup
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
  # while True:
  run()
  time.sleep(1)

