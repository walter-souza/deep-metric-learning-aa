import subprocess

# ['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '128', '--seed', '0'],
# ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '128', '--seed', '0'],
# ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '128', '--seed', '0'],
# ['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '64', '--seed', '0'],
# ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '64', '--seed', '0'],
# ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '64', '--seed', '0'],

# ['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '128', '--seed', '1'],
# ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '128', '--seed', '1'],
# ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '128', '--seed', '1'],

# ['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '64', '--seed', '1'],
# ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '64', '--seed', '1'],
# ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '64', '--seed', '1'],

# ['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '128', '--seed', '2'],
# ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '128', '--seed', '2'],
# ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '128', '--seed', '2'],

# ['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '64', '--seed', '2'],
# ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '64', '--seed', '2'],
# ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '64', '--seed', '2']

program_list = [['python', 'dml_train.py', '--embedding_size', '256', '--token_size', '256', '--batch_size', '48', '--n_epochs', '6', '--seed', '0'],
                ['python', 'dml_train.py', '--embedding_size', '128', '--token_size', '256', '--batch_size', '48', '--n_epochs', '6', '--seed', '0'],
                ['python', 'dml_train.py', '--embedding_size',  '64', '--token_size', '256', '--batch_size', '48', '--n_epochs', '6', '--seed', '0'],]

for program in program_list:
    print(program)
    subprocess.call(program)
    print("Finished:" + ' '.join(program))