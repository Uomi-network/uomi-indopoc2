import json
import os
import sys

# Configurations
############################################

SNAPSHOT_FOLDER = os.getenv('SNAPSHOT_FOLDER', './snapshots')

# Functions
############################################

def list_checks():
  if not os.path.exists(SNAPSHOT_FOLDER):
    print('❌ No snapshots found')
    return []

  snapshots = []
  for snapshot in os.listdir(SNAPSHOT_FOLDER + '/db_13'):
    snapshots.append(snapshot)
  return snapshots

def get_check(check_key):
  data = {}
  with open(f'{SNAPSHOT_FOLDER}/db_13/{check_key}', 'r') as f:
    data = json.load(f)
  return data

def get_execution(key, node):
  data = {}
  with open(f'{SNAPSHOT_FOLDER}/db_{node}/{key}', 'r') as f:
    data = json.load(f)
  return data

def main():
  checks = list_checks()
  error_rates = []
  position_rates = []
  max_position_changed = 0
  for check in checks:
    check_data = get_check(check)
    execution_data = get_execution(check_data['key'], check_data['executed_by'])
    check_result_str = '✅' if check_data['check_result'] else '❌'
    print(f'{check_result_str} Check: {check_data["key"]}')
    print(f'- Checked by: {check_data["checked_by"]}')
    print(f'- Executed by: {check_data["executed_by"]}')

    # Analyze the tokens
    errors = []
    positions = []
    for i, check_token in enumerate(check_data['check_data']):
      execution_token = execution_data['execution_data'][i]
      check_token_prob = check_token['prob'] if check_token['prob'] else 9999
      execution_token_prob = execution_token['prob']
      execution_token_pos = -1
      for j, token in enumerate(execution_token['top_k']):
        if token['id'] == execution_token['id']:
          execution_token_pos = j
          break
      check_token_id = check_token['id']
      # Calculate the error rate
      error_rate = abs(check_token_prob - execution_token_prob)
      errors.append(error_rate)
      # Calculate the position of check_token_id in execution_token['top_k']
      check_token_pos = -1
      for j, token in enumerate(check_token['top_k']):
        if token['id'] == check_token_id:
          check_token_pos = j
          break
      if check_token_pos != execution_token_pos:
        print(f'  -- Position changed on index {i} from {execution_token_pos} to {check_token_pos}')
      position_change = abs(check_token_pos - execution_token_pos)
      positions.append(position_change)
      if position_change > max_position_changed:
        max_position_changed = position_change

      
    error_rate = sum(errors) / len(errors)
    error_rates.append(error_rate)
    print(f'- Error rate: {error_rate}')

    position_rate = sum(positions) / len(positions)
    position_rates.append(position_rate)
    print(f'- Position rate: {position_rate}')
  
  print(f'📈 Media errore su probabilità del token totale: {sum(error_rates) / len(error_rates)}')
  print(f'📈 Media errore sulla posizione del token totale: {sum(position_rates) / len(position_rates)}')
  print(f'📈 Massima differenza di posizione del token: {max_position_changed}')
  
  position_changed = len([rate for rate in position_rates if rate > 0])
  print(f'📈 Controlli con token che ha cambiato posizione almeno una volta: {position_changed}')


# Main
############################################

if __name__ == '__main__':
  main()
  print('*'*50)
