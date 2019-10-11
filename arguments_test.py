import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-laplace', help="adds laplace smoothing", action="store_true")
parser.add_argument('cipher', type=str)
args = parser.parse_args()

if args.laplace:
    print(6)

if args.cipher == 'cipher1':
    print('cipher1 was selected')
