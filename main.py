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

SIMULATION_MODE = False

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
if not SIMULATION_MODE:
  model_name = "casperhansen/mistral-small-24b-instruct-2501-awq"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)
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
  if SIMULATION_MODE:
    time.sleep(2)
    return json.dumps({
      "key": key,
      "output": prompt,
      "execution_data": [],
      "executed_by": NODE_ID,
      "executed_in": 2
    })

  time_start = time.time()
  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

  execution_data = []
  first_new_token_id = None

  for step in range(MAX_NEW_TOKENS):
    print(f"Step execute_inference {step + 1}/{MAX_NEW_TOKENS}")
    # Forward pass to get raw logits
    time_start = time.time()
    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    print(f"-- time for next_token_logits: {time.time() - time_start}")

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
    time_start = time.time()
    probs = F.softmax(next_token_logits, dim=-1)  # shape: [1, vocab_size]
    print(f"-- time for probs: {time.time() - time_start}")

    # Print the top-k tokens by probability
    time_start = time.time()
    top_probs, top_indices = probs.topk(TOP_K_EXECUTION, dim=-1)
    print(f"-- time for top-k: {time.time() - time_start}")
    time_start = time.time()
    # execution_data_top_k = []
    # for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
    #   token_str = tokenizer.decode([idx.item()])
    #   execution_data_top_k.append({
    #     "str": token_str,
    #     "prob": prob.item(),
    #     "id": idx.item()
    #   })
    execution_data_top_k = []
    for idx in top_indices[0]:
      token_str = tokenizer.decode([idx.item()])
      prob = probs[0, idx].item()
      execution_data_top_k.append({
        "str": token_str,
        "prob": prob,
        "id": idx.item()
      })
    print(f"-- time for top-k loop: {time.time() - time_start}")

    # GREEDY selection instead of sampling
    # This ensures full determinism.
    time_start = time.time()
    next_token_id = top_indices.select(-1, torch.multinomial(top_probs, num_samples=1).item()).unsqueeze(0)
    selected_token_id = next_token_id.item()
    selected_token_str = tokenizer.decode([selected_token_id])
    selected_token_prob = probs[0, selected_token_id].item()
    print(f"-- time for greedy selection: {time.time() - time_start}")

    # Append the chosen token
    time_start = time.time()
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    print(f"-- time for input_ids concat: {time.time() - time_start}")

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

# This function execute the check of an inference and return the result
# NOTE: Checking an inference means to take the output of the inference and check for each token if its probability is in the TOK_K_CHECK of the new inference
def execute_check(inference):
  if SIMULATION_MODE:
    time.sleep(2)
    return json.dumps({
      "key": "key",
      "check_result": True,
      "check_data": [],
      "checked_by": NODE_ID,
      "checked_in": 2,
      "executed_by": NODE_ID,
      "executed_in": 2
    })

  time_start = time.time()

  inference = json.loads(inference)
  prompt = r_prompts_db.get(inference["key"]).decode('utf-8')

  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
  check_data = []
  check_result = True

  for step in range(MAX_NEW_TOKENS):
    print(f"Step execute_check {step + 1}/{MAX_NEW_TOKENS}")

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
      cutoff_idx = cutoff_idx.clamp(max=sorted_probs.size(-1) - 1)
      sorted_logits_1d[cutoff_idx + 1:] = float('-inf')

      # Scatter back
      next_token_logits = torch.full_like(next_token_logits, float('-inf'))
      next_token_logits[0].scatter_(0, sorted_indices_1d, sorted_logits_1d)

    # Convert to probabilities
    probs = F.softmax(next_token_logits, dim=-1)  # shape: [1, vocab_size]

    # Take the current token from the inference output and check it's probability on the model
    check_data_top_k = []
    current_token_prob = None
    # calculate step_without_prompt by removing the prompt from the inference_output_tokens
    current_token_id = inference["output_tokens"][step]
    current_token_str = tokenizer.decode([current_token_id])
    top_probs, top_indices = probs.topk(TOK_K_CHECK + 5, dim=-1)
    
    # top_probs = top_probs[0]  # Remove batch dimension
    # top_indices = top_indices[0]  # Remove batch dimension

    # # Check if current token is in top-k (for determining check_result)
    # is_in_top_k = (top_indices[:TOK_K_CHECK] == current_token_id).any()
    # current_token_prob = float(probs[0, current_token_id].item()) if is_in_top_k else None

    # # Always create the check_data_top_k list with top_k + 5 tokens
    # check_data_top_k = []
    # for i in range(len(top_indices)):
    #   idx = top_indices[i].item()
    #   prob = top_probs[i].item()
    #   # Only decode tokens once - this is expensive
    #   token_str = tokenizer.decode([idx])
    #   check_data_top_k.append({
    #     "str": token_str,
    #     "prob": prob,
    #     "id": idx
    #   })

    index = 0
    # for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
    #   index += 1
    #   token_str = tokenizer.decode([idx.item()])
    #   check_data_top_k.append({
    #     "str": token_str,
    #     "prob": prob.item(),
    #     "id": idx.item()
    #   })
    #   # print(f"   {rank}. '{token_str}' -> prob={prob.item():.6f}")
    #   if idx == current_token_id and index <= TOK_K_CHECK:
    #     current_token_prob = float(prob.item())
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

    if current_token_prob is None:
      check_result = False
      check_data.append({
        "str": current_token_str,
        "prob": current_token_prob,
        "id": current_token_id,
        "top_k": check_data_top_k
      })
      print(f"❌ Current token: '{current_token_str}' -> not found in top-{TOK_K_CHECK}")
      break

    # GREEDY selection instead of sampling
    # This ensures full determinism.
    selected_token_str = tokenizer.decode([current_token_id])
    selected_token_prob = probs[0, current_token_id].item()

    # Append the chosen token
    next_token_id = torch.tensor([[current_token_id]]).to(device)
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Append the check result
    check_data.append({
      "str": current_token_str,
      "prob": current_token_prob,
      "id": current_token_id,
      "top_k": check_data_top_k
    })

  result = {
    "key": inference["key"],
    "check_result": check_result,
    "check_data": check_data,
    "checked_by": NODE_ID,
    "checked_in": time.time() - time_start,
    "executed_by": inference["executed_by"],
    "executed_in": inference["executed_in"]
  }
  print(result)
  return json.dumps(result)

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
        # BACKUP: using multiple threads
        # elif not prompts_runned_one:
        #   print("Executing inference: " + str(key))
        #   inferences_to_run = (prompt, key)
        #   prompts_runned_one = True
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
    check_runned_one = False
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
          # BACKUP: using multiple threads
          # elif not check_runned_one:
          #   print("Executing check: " + str(check_key))
          #   inference = node_inferences_db.get(key).decode('utf-8')
          #   check_to_run = (inference, check_key)
          #   check_runned_one = True
          elif not check_runned_one:
            print("Executing check: " + str(check_key))
            inference = node_inferences_db.get(key).decode('utf-8')
            check_result = execute_check(inference)
            # r_checks_db.set(check_key, check_result)
            check_runned_one = True
          else:
            print("Remaining op check: " + str(remaining))
            remaining += 1
            break
        if not check_runned_one:
          completed[node] = True
          print("✅ Node " + str(node) + " completed the checks.")
        if check_runned_one:
          break

    # BACKUP: using multiple threads
    # # Run the inference and check on two different threads
    # def run_inference(prompt, key):
    #   print("🤖 Executing inference: " + str(key))
    #   result = execute_inference(prompt, key)
    #   r_node_inferences_db.set(key, result)
    # def run_check(inference, key):
    #   print("🤖 Executing check: " + str(key))
    #   result = execute_check(inference)
    #   r_checks_db.set(key, result)
    # threads = []
    # if inferences_to_run:
    #   threads.append(threading.Thread(target=run_inference, args=inferences_to_run))
    # if check_to_run:
    #   threads.append(threading.Thread(target=run_check, args=check_to_run))
    # for thread in threads:
    #   thread.start()
    # for thread in threads:
    #   thread.join()
    # print("🤖 Inference and check completed.")

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
  while True:
    run()
    time.sleep(1)

