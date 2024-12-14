import os
from datasets import load_dataset
import subprocess

# Dataset names and corresponding URLs
DATASETS = {
    "MathInstruct": "https://huggingface.co/datasets/TIGER-Lab/MathInstruct",
    "python_code_instructions": "https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca",
    "dolly": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
    "billsum": "https://huggingface.co/datasets/FiscalNote/billsum",
    "commitpackft": "https://huggingface.co/datasets/chargoddard/commitpack-ft-instruct",
    "NuminaMath-CoT": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
    "PubMedQA": "https://huggingface.co/datasets/qiaojin/PubMedQA",
    "MetaMathQA": "https://huggingface.co/datasets/meta-math/MetaMathQA",
    "evol_instruct": "https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K",
    "cnn_dailymail": "https://huggingface.co/datasets/abisee/cnn_dailymail",
    "xsum": "https://huggingface.co/datasets/EdinburghNLP/xsum",
    "meetingbank": "https://huggingface.co/datasets/huuuyeah/meetingbank",
}

# Base directory for dataset folders
ROOT_FOLDER = "data"

# Create base directory if it doesn't exist
os.makedirs(ROOT_FOLDER, exist_ok=True)

for dataset_name, url in DATASETS.items():
    # Create dataset folder
    dataset_dir = os.path.join(ROOT_FOLDER, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory: {dataset_dir}")
    else:
        print(f"Directory already exists: {dataset_dir}")
        continue

    # Download dataset
    try:
        print(f"Downloading dataset: {dataset_name} from {url}")
        dataset = load_dataset(url, cache_dir=dataset_dir)
        print(f"Downloaded dataset: {dataset_name}")
    except Exception as e:
        print(f"Failed to download dataset {dataset_name}: {e}")
        continue

    # Preprocess dataset
    preprocess_script = "data_utils/gpt_load_dataset.py"
    if os.path.exists(preprocess_script):
        try:
            print(f"Preprocessing dataset: {dataset_name}")
            subprocess.run([
                "python", preprocess_script, \
                "--dataset", dataset_name, \
                "--key", 'text', \
                "--root_folder", ROOT_FOLDER
            ], check=True)
            print(f"Preprocessed dataset: {dataset_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to preprocess dataset {dataset_name}: {e}")
    else:
        print(f"Preprocessing script not found: {preprocess_script}")
