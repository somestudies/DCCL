import argparse

parser = argparse.ArgumentParser('Parser for Causal Model')

# -- Basic -- #
parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 2022)')
parser.add_argument('--dataset', type=str, default='yelp', help='dataset name')

# -- Training -- #
parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--dccl_int_weight', type=float, default=1e-1, help='dccl interest contrastive loss')
parser.add_argument('--dccl_conf_weight', type=float, default=1e-1, help='dccl conformity contrastive loss')
parser.add_argument('--gpus', type=str, default='0', help='gpus')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--pop_coeff', type=float, default=1.0, help='dccl popularity function coefficient')
parser.add_argument('--score_coeff', type=float, default=1.0, help='dccl score coefficient')

args, _ = parser.parse_known_args()

print('#' * 132)
print(args)
print('#' * 132)
