import os
import tarfile
import subprocess

# Dataset names and corresponding URLs
DATASETS = {
    "MathInstruct": "TIGER-Lab/MathInstruct",
    "python_code_instructions": "iamtarun/python_code_instructions_18k_alpaca",
    "dolly": "databricks/databricks-dolly-15k",
    "billsum": "FiscalNote/billsum",
    "commitpackft": "chargoddard/commitpack-ft-instruct",
    "NuminaMath-CoT": "AI-MO/NuminaMath-CoT",
    "PubMedQA": ("qiaojin/PubMedQA", "pqa_artificial"),
    "MetaMathQA": "meta-math/MetaMathQA",
    "evol_instruct": "ise-uiuc/Magicoder-Evol-Instruct-110K",
    "cnn_dailymail": ("abisee/cnn_dailymail", "1.0.0"),
    "xsum": "EdinburghNLP/xsum",
    "meetingbank": "huuuyeah/meetingbank",
}

# Base directory for dataset folders
ROOT_FOLDER = "data"
GIT_REPO_URL = "https://huggingface.co/Somoku/tesatad-tcafitra"
TAR_FILE_NAME = "ft_dataset.tar.gz"

def install_git_lfs():
    try:
        subprocess.run(["git", "lfs", "install"], cwd=ROOT_FOLDER, check=True)
        print("Git LFS installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during git lfs install: {e}")

def clone_repository():
    subprocess.run(["git", "clone", GIT_REPO_URL], cwd=ROOT_FOLDER, check=True)
    print("Repository cloned successfully.")

def extract_tar_file():
    tar_file_path = os.path.join(ROOT_FOLDER, "tesatad-tcafitra", "ft_dataset.tar.gz")
    if os.path.exists(tar_file_path):
        try:
            subprocess.run(["mv", tar_file_path, ROOT_FOLDER], check=True)
            print("Tar file moved successfully.")
        except Exception as e:
            print(f"Error during tar file move: {e}")
    tar_file_path = os.path.join(ROOT_FOLDER, TAR_FILE_NAME)
    if os.path.exists(tar_file_path):
        try:
            with tarfile.open(tar_file_path, "r:gz") as tar:
                tar.extractall(path=ROOT_FOLDER)
            print("Dataset extracted successfully.")
            # subprocess.run(["mv", os.path.join(ROOT_FOLDER, "finetune-datasets", r"*"), ROOT_FOLDER], check=True)
            # subprocess.run(["rm", "-rf", os.path.join(ROOT_FOLDER, "finetune-datasets")], check=True)
            # print("Dataset moved successfully.")
        except Exception as e:
            print(f"Error during tar extraction: {e}")
    else:
        print(f"Tar file {TAR_FILE_NAME} not found in {ROOT_FOLDER}.")

# Create base directory if it doesn't exist
os.makedirs(ROOT_FOLDER, exist_ok=True)

if not os.path.exists(os.path.join(ROOT_FOLDER, TAR_FILE_NAME)):
    # Install Git LFS
    install_git_lfs()

    # Clone repository
    retries = 10
    for i in range(retries):
        try:
            clone_repository()
            break
        except Exception as e:
            print(f"Failed to clone repository: {e}")
            if i < retries - 1:
                print(f"Retrying... Attempt {i + 1}")
            else:
                print("Maximum retries reached. Exiting...")
                exit(-1)

# Extract tar file
extract_tar_file()

for dataset_name, _ in DATASETS.items():
    # Create dataset folder
    dataset_dir = os.path.join(ROOT_FOLDER, dataset_name)

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
