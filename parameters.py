import argparse

def training_parameters(parser):
    parser.add_argument('--dataset',          default='imdb62',    type=str)
    parser.add_argument('--loss',             default='arcface',   type=str)
    parser.add_argument('--n_epochs',         default='20',        type=int)
    parser.add_argument('--embedding_size',   default='256',       type=int)
    parser.add_argument('--token_size',       default='128',       type=int)
    parser.add_argument('--batch_size',       default='64',        type=int)
    parser.add_argument('--seed',             default='21',        type=int)
    return parser