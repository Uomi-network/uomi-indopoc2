import sys
import time
import os
import redis
import json

# Configurations
############################################

SNAPSHOT_FOLDER = os.getenv('SNAPSHOT_FOLDER', './snapshots')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASS = os.getenv('REDIS_PASS', '')
REDIS_PORT = 6379
REDIS_NODES_DB = 0
REDIS_ASSIGNMENTS_DB = 12
REDIS_CHECKS_DB = 13
REDIS_COMPLETITION_DB = 14
REDIS_PROMPTS_DB = 15

NODES = [
  1, # RTX 4090
  2, # RTX A6000
  3, # H100 XMS
  4, # L40S
  5, # A100 SXM
  6, # LOCAL
]

# Functions
############################################

def snapshot_db(db, override=False):
  snapshot_path = f'{SNAPSHOT_FOLDER}/db_{db}'
  os.makedirs(snapshot_path, exist_ok=True)
  try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db, password=REDIS_PASS)
  except:
    print(f'❌ Error connecting to Redis DB: {db}')
    return
  # Store single datas as files
  keys = r.keys()
  print(f'  - Keys: {len(keys)}')
  for key in keys:
    key = key.decode('utf-8')
    # Skip key if already stored
    if not override and os.path.exists(f'{snapshot_path}/{key}'):
      continue
    with open(f'{snapshot_path}/{key}', 'w') as f:
      f.write(r.get(key).decode('utf-8'))

def snapshot():
  snapshot_key = int(time.time())
  print(f'🤖 Snapshot start')

  for db in [REDIS_NODES_DB, REDIS_CHECKS_DB, REDIS_COMPLETITION_DB, REDIS_PROMPTS_DB, REDIS_ASSIGNMENTS_DB]:
    print(f'- DB: {db}')
    snapshot_db(db, db != REDIS_PROMPTS_DB and db != REDIS_CHECKS_DB and db != REDIS_ASSIGNMENTS_DB)
  for node in NODES:
    print(f'- Node: {node}')
    snapshot_db(node)

  print(f'✅ Snapshot done!')

def recap():
  print(f'📊 Recap')
  # Calculate total prompts count (number of keys in the db_15 snapshot)
  total_prompts = 0
  for key in os.listdir(f'{SNAPSHOT_FOLDER}/db_15'):
    total_prompts += 1
  print(f'- Total prompts: {total_prompts}')
  # Calculate number of valid and invalid checks (reading files in db_13 snapshot)
  total_check_pass = 0
  total_check_fail = 0
  for key in os.listdir(f'{SNAPSHOT_FOLDER}/db_13'):
    with open(f'{SNAPSHOT_FOLDER}/db_13/{key}', 'r') as f:
      f_json = json.loads(f.read())
      if f_json['check_result']:
        total_check_pass += 1
      else:
        print(f'❌ Check failed: {f_json["key"]}')
        total_check_fail += 1
  print(f'- Total checks passed: {total_check_pass}')
  print(f'- Total checks failed: {total_check_fail}')
  # Calculate stats for each node
  for node in NODES:
    print(f'- Node: {node}')
    # Read remaining activities from db_14 snapshot
    remaining_activities = None
    if os.path.exists(f'{SNAPSHOT_FOLDER}/db_14/{node}'):
      with open(f'{SNAPSHOT_FOLDER}/db_14/{node}', 'r') as f:
        remaining_activities = int(f.read())
    print(f'  - Remaining activities: {remaining_activities}')
  
  print(f'✅ Recap done!')

# Main
############################################

if __name__ == '__main__':
  while True:
    print(' ')
    snapshot()
    print(' ')
    recap()
    print('*'*50)
    time.sleep(30)
