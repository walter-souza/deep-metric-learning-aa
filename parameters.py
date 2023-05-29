import argparse

# python dml_train.py --dataset imdb62 --loss arcface --embedding_size 32

def training_parameters(parser):
    parser.add_argument('--dataset',          default='imdb62',                    type=str)
    parser.add_argument('--loss',             default='arcface',                   type=str)
    parser.add_argument('--base_model',       default='distilbert-base-uncased',   type=str)
    parser.add_argument('--n_epochs',         default=20,                          type=int)
    parser.add_argument('--embedding_size',   default=256,                         type=int)
    parser.add_argument('--token_size',       default=128,                         type=int)
    parser.add_argument('--batch_size',       default=64,                          type=int)
    parser.add_argument('--seed',             default=21,                          type=int)
    parser.add_argument('--test_size',        default=0.25,                        type=float)
    return parser