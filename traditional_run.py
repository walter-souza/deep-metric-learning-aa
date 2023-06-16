import subprocess

# ['python', 'traditional_train.py', '--n_epochs', '6', '--token_size', '128', '--seed', '0'],
# ['python', 'traditional_train.py', '--n_epochs', '6', '--token_size',  '64', '--seed', '0'],
# ['python', 'traditional_train.py', '--n_epochs', '6', '--token_size', '128', '--seed', '1'],
# ['python', 'traditional_train.py', '--n_epochs', '6', '--token_size',  '64', '--seed', '1'],
# ['python', 'traditional_train.py', '--n_epochs', '6', '--token_size', '128', '--seed', '2'],

program_list = [
                ['python', 'traditional_train.py', '--n_epochs', '6', '--token_size',  '64', '--seed', '2'],]

for program in program_list:
    print(program)
    subprocess.call(program)
    print("Finished:" + ' '.join(program))