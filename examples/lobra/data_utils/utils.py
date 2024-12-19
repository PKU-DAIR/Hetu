import io
import os
import json
import pyarrow.parquet as pq

def _load_dataset_to_json(dataset_name, root_folder='data', override=False):
    """Load dataset from parquet file and save as json file."""
    json_file = f'{root_folder}/{dataset_name}/{dataset_name}.json'
    if os.path.exists(json_file) and not override:
        return json_file
    parquet_files = []
    # 遍历f'{root_folder}/{dataset_name}'目录下的所有文件
    for _, _, files in os.walk(f'{root_folder}/{dataset_name}'):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(file)
    
    json_data = ''
    for parquet_file in parquet_files:
        tabel = pq.read_table(f'{root_folder}/{dataset_name}/{parquet_file}')
        df = tabel.to_pandas()
        json_data += df.to_json(orient='records', lines=True)
    
    f_dirname = os.path.dirname(json_file)
    if f_dirname != "":
        os.makedirs(f_dirname, exist_ok=True)
    f = open(json_file, 'w')
    f.write(json_data)
    f.close()
    return json_file

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or a dictionary to a file in json format.
    
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f, mode="r"):
    """Load a json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
