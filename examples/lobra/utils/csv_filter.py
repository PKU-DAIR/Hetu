import os
import argparse
import pandas as pd

def csv_filter(args):
    """
    Filter a csv file by columns in args.filter_column
    """
    if not os.path.exists(args.input_path):
        print(f"Input file not found: {args.input_path}, maybe OOM happened.")
        return
    if not args.output_path:
        args.output_path = args.input_path + ".raw"
    if not os.path.isfile(args.input_path) or os.stat(args.input_path).st_size == 0:
        print("Input file is empty")
        return
    d = pd.read_csv(args.input_path, usecols=args.filter_column)
    d.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default='', help="input csv file path"
    )
    parser.add_argument(
        "--output_path", type=str, default='', help="output csv file path"
    )
    parser.add_argument(
        "--filter_column", type=str, default='', nargs='+', help="column to filter"
    )
    args = parser.parse_args()
    csv_filter(args)
    