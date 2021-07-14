# Andreas Goulas <goulasand@gmail.com>

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('csv0', help='first csv file')
    parser.add_argument('csv1', help='second csv file')
    parser.add_argument('out', help='save path')
    parser.add_argument('--tsv', action='store_true', help='whether the files are stored as tsv')
    args = parser.parse_args()

    if args.tsv:
        df0 = pd.read_csv(args.csv0, sep='\t')
        df1 = pd.read_csv(args.csv1, sep='\t')
    else:
        df0 = pd.read_csv(args.csv0)
        df1 = pd.read_csv(args.csv1)

    out_df = pd.concat([df0, df1], axis=1)
    if args.tsv:
        out_df.to_csv(args.out, index=None, sep='\t')
    else:
        out_df.to_csv(args.out, index=None)

if __name__ == '__main__':
    main()