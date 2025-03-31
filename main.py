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
TOK_K_CHECK = 15
BATCH_SIZE_CHECK = 5
BATCH_SIZE_INFERENCE = 5

# Enable or disable KV caching (you can toggle this flag)
USE_KV_CACHE = True

NODES = [
  1, # RTX 4090
  2, # RTX A6000
  3, # H100 SXM
  4, # L40S
  5, # A100 SXM
]

# Model configuration class
class ModelConfig:
    def __init__(self, name, deterministic=False):
        self.name = name
        self.deterministic = deterministic

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

# Models configuration - similar to the original code's self.models_config
models_config = {
    model_name: ModelConfig(name=model_name, deterministic=False)
}

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
    time_start = time.time()
    batch_size = len(batch_prompts)
    
    # Set deterministic settings
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all prompts with padding
    inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_lengths = [mask.sum().item() for mask in attention_mask]

    # Setup generation parameters
    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": True,
        "use_cache": True,
        "top_k": TOP_K_EXECUTION,
        "top_p": TOP_P,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    # Generate outputs
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )

    # Process outputs
    generated_sequences = outputs.sequences
    scores = outputs.scores  # list of (batch_size, vocab_size) tensors

    results = []
    for i, key in enumerate(batch_keys):
        input_ids_i = input_ids[i]
        prompt_length = attention_mask.size(1) #prompt_lengths[i]
        generated_sequence = generated_sequences[i]
        generated_tokens = generated_sequence[prompt_length:].tolist()

        # Extract execution data (top_k for each generated token)
        execution_data = []
        for token_idx in range(len(generated_tokens)):
            if token_idx >= len(scores):
                break  # Handle cases where scores are missing
            step_scores = scores[token_idx][i]
            probs = F.softmax(step_scores, dim=-1)
            top_probs, top_indices = probs.topk(TOP_K_EXECUTION)

            # Find selected token's probability and rank
            selected_token_id = generated_tokens[token_idx]
            selected_token_prob = None
            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                if idx.item() == selected_token_id:
                    selected_token_prob = prob.item()
                    break

            # Collect top_k data
            top_k_list = []
            for prob, idx in zip(top_probs, top_indices):
                top_k_list.append({
                    "str": tokenizer.decode([idx.item()]),
                    "prob": prob.item(),
                    "id": idx.item()
                })

            execution_data.append({
                "str": tokenizer.decode([selected_token_id]),
                "prob": selected_token_prob if selected_token_prob is not None else probs[selected_token_id].item(),
                "id": selected_token_id,
                "top_k": top_k_list
            })

        # Decode the full output
        generated_tokens = [token for token in generated_tokens if token not in tokenizer.all_special_tokens]
        generated_tokens = [token for token in generated_tokens if token not in [151643,151646]]

        output = tokenizer.decode(generated_sequence, skip_special_tokens=True)

        result = {
            "key": key,
            "output": output,
            "output_tokens": generated_tokens,
            "execution_data": execution_data,
            "executed_by": NODE_ID,
            "executed_in": time.time() - time_start,
            "full_sequence_length": generated_sequence.size(0)
        }
        results.append(json.dumps(result))
        print(f"✅ Inference {i} completed")

    print(f"✅ Batch inference completed: {time.time() - time_start}s for {batch_size} prompts")
    return results
def execute_batch_checks(batch_checks):
    print(batch_checks)
    time_start = time.time()
    inferences = [json.loads(inference) for inference in batch_checks]
    batch_size = len(inferences)
    
    # Set deterministic settings
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get prompts and generated tokens
    prompts = [r_prompts_db.get(inference["key"]).decode('utf-8') for inference in inferences]
    generated_tokens_list = [inference["output_tokens"] for inference in inferences]
    for arr in generated_tokens_list:
        print(f"{tokenizer.decode(arr)=}")

    # Prepare full input sequences (prompt + generated tokens)
    full_input_ids = []
    prompt_lengths = []
    generated_lengths = []
    for prompt, generated_tokens in zip(prompts, generated_tokens_list):
        # Tokenize prompt
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = prompt_inputs.input_ids[0].to(device)
        prompt_length = prompt_ids.size(0)
        generated_length = len(generated_tokens)
        
        # Combine prompt and generated tokens
        generated_ids = torch.tensor(generated_tokens, dtype=torch.long, device=device)
        full_ids = torch.cat([prompt_ids, generated_ids])
        full_input_ids.append(full_ids)
        prompt_lengths.append(prompt_length)
        generated_lengths.append(generated_length)

    # Pad sequences to max length
    max_length = max(len(ids) for ids in full_input_ids)
    padded_input_ids = torch.stack([
        torch.cat([ids, torch.full((max_length - len(ids),), tokenizer.pad_token_id, dtype=torch.long, device=device)])
        for ids in full_input_ids
    ])
    attention_mask = (padded_input_ids != tokenizer.pad_token_id).long().to(device)
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(padded_input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Process each inference
    results = []
    for i in range(batch_size):
        prompt_length = prompt_lengths[i]
        generated_length = generated_lengths[i]
        check_data = []
        valid = True

        for step in range(generated_length):
            logits_pos = prompt_length - 1 + step
            if logits_pos >= logits.size(1) - 1:
                break  # Beyond sequence length due to padding

            current_logits = logits[i, logits_pos, :]
            current_token_id = generated_tokens_list[i][step]

            # Get top-k tokens and probabilities
            probs = F.softmax(current_logits, dim=-1)
            top_probs, top_indices = probs.topk(TOK_K_CHECK)

            # Check if current token is in top-k
            found = False
            current_token_prob = None

            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                if idx.item() == current_token_id:
                    current_token_prob = prob.item()
                    found = True
                    break

            # Collect check data
            top_k_list = []
            for prob, idx in zip(top_probs, top_indices):
                top_k_list.append({
                    "str": tokenizer.decode([idx.item()]),
                    "prob": prob.item(),
                    "id": idx.item()
                })

            check_data.append({
                "str": tokenizer.decode([current_token_id]),
                "prob": current_token_prob if found else probs[current_token_id].item(),
                "id": current_token_id,
                "top_k": top_k_list
            })

            if not found:
                valid = False
                print(f"check failed, {check_data=}")

        result = {
            "key": inferences[i]["key"],
            "check_result": valid,
            "check_data": check_data,
            "checked_by": NODE_ID,
            "checked_in": time.time() - time_start,
            "executed_by": inferences[i]["executed_by"],
            "executed_in": inferences[i]["executed_in"]
        }
        results.append(json.dumps(result))
        print(f"✅ Check {i} completed")


    print(f"✅ Batch check completed: {time.time() - time_start}s for {batch_size} inferences")
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