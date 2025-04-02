import numpy as np
from typing import List, Tuple, Optional

def build_fake_batch_and_len(
    fake_seqlens: List[int],
    pad_token_id: int,
    fake_fill_value: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Build a fake batch according to given seqlens of each sample, sorted by seqlens.

    Args:
        fake_seqlens (List[int]): list of seqlens of each sample.
        pad_token_id (int): pad token id of the tokenizer.
        fake_fill_value (int, optional): fill value for the fake batch. Defaults to 0.

    Returns:
        Tuple[np.ndarray, List[int]]: Tuple of fake batch and sorted seqlens.
            The fake batch is padded to the maximum of `fake_seqlens`.
    """

    # 对fake_seqlens进行排序
    sorted_seqlens = sorted(fake_seqlens)
    max_len = sorted_seqlens[-1]
    result = np.full((len(sorted_seqlens), max_len), pad_token_id, dtype=int)
    assert pad_token_id != fake_fill_value, \
        f"fake_fill_value {fake_fill_value} conflicts with pad token id."
    for i, seq_len in enumerate(sorted_seqlens):
        result[i, :seq_len] = fake_fill_value
    return result, sorted_seqlens

def convert_parquet_to_json(
    parquet_file: str,
    json_file: Optional[str] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
):
    """Convert parquet file to json file.

    Args:
        parquet_file (str): path to the parquet file.
        json_file (str): path to the json file.
        columns (List[str], optional): list of columns to be converted.
        chunksize (int, optional): chunk size for reading the parquet file. Defaults to 100000.
    """
    import pandas as pd
    import json
    from tqdm import tqdm

    if json_file is None:
        json_file = parquet_file.replace('.parquet', '.json')

    reader = pd.read_parquet(parquet_file, columns=columns, chunksize=chunksize)
    with open(json_file, 'w') as f:
        for chunk in tqdm(reader):
            for _, row in chunk.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')
    return json_file

__all__ = [
    'build_fake_batch_and_len',
    'convert_parquet_to_json',
]