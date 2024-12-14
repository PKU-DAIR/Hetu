import argparse
import numpy as np
import pandas as pd
from data_utils import load_dataset, get_length_list

def analyze_datasets(datasets):
    dataset_len_map = {name : get_length_list(dataset) for name, dataset in datasets.items()}
    df = pd.DataFrame(dataset_len_map)
    skew_dict = df.skew(axis=0).to_dict()
    kurt_dict = df.kurt(axis=0).to_dict()

    def get_length_distribution(length_list):
        seq_len_num_distribution = {}
        for length in length_list:
            padded_len = max(2 ** length.bit_length(), 256)
            seq_len_num_distribution[padded_len] = seq_len_num_distribution.get(padded_len, 0) + 1
        seq_len_num_distribution = dict(sorted(seq_len_num_distribution.items(), key=lambda x: x[0]))
        for seq_len, num in seq_len_num_distribution.items():
            seq_len_num_distribution[seq_len] = num / len(length_list)
        return seq_len_num_distribution
    
    length_dict = {name : get_length_distribution(length_list) for name, length_list in dataset_len_map.items()}
    avg_seq_len_dict = {name : np.mean(length_list) for name, length_list in dataset_len_map.items()}
    return skew_dict, kurt_dict, avg_seq_len_dict, length_dict
        
def get_skewness_of_datasets(datasets):
    dataset_len_map = {name : get_length_list(dataset) for name, dataset in datasets.items()}
    df = pd.DataFrame(dataset_len_map)
    return df.skew(axis=0).to_dict()
    
def get_kurtosis_of_datasets(datasets):
    dataset_len_map = {name : get_length_list(dataset) for name, dataset in datasets.items()}
    df = pd.DataFrame(dataset_len_map)
    return df.kurt(axis=0).to_dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=str, default='alpaca/alpaca'
    )
    parser.add_argument(
        "--keys", type=str, default='text'
    )
    parser.add_argument(
        "--vocab_file", type=str, default=None
    )
    parser.add_argument(
        "--merge_file", type=str, default=None
    )
    parser.add_argument(
        "--root_folder", type=str, default='data'
    )
    parser.add_argument(
        "--cache_path", type=str, default=None
    )
    
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    keys = args.keys.split(',')
    datasets_map = {}
    for dataset, key in zip(datasets, keys):
        datasets_map[dataset] = load_dataset(dataset, key, vocab_file=args.vocab_file,
                                             merge_file=args.merge_file, cache_path=args.cache_path,
                                             root_folder=args.root_folder)
    skew_dict, kurt_dict, avg_seq_len_dict, length_dict = analyze_datasets(datasets_map)
    print("********************************")
    print("Skewness of datasets:")
    print(skew_dict)
    print("********************************")
    print("Kurtosis of datasets:")
    print(kurt_dict)
    print("********************************")
    print("Avg seq len of datasets:")
    print(avg_seq_len_dict)
    print("********************************")
    print("Seq len distribution of datasets:")
    print(length_dict)
    print("********************************")
