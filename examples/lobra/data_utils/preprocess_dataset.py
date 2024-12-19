import json
import argparse
import os
from utils import jload, jdump, _load_dataset_to_json
from prompt_template import AlpacaInstructTemplate, StackExchangedPairedTemplate, \
                            SummarizeTemplate, StackExchangedPairedwithContextTemplate

root_folder = "data"

dataset_to_template = {
    "MathInstruct": AlpacaInstructTemplate,
    "python_code_instructions": AlpacaInstructTemplate,
    "dolly": AlpacaInstructTemplate,
    "billsum": SummarizeTemplate,
    "commitpackft": AlpacaInstructTemplate,
    "NuminaMath-CoT": StackExchangedPairedTemplate,
    "PubMedQA": StackExchangedPairedwithContextTemplate,
    "MetaMathQA": StackExchangedPairedTemplate,
    "evol_instruct": AlpacaInstructTemplate,
    "cnn_dailymail": SummarizeTemplate,
    "xsum": SummarizeTemplate,
    "meetingbank": SummarizeTemplate,
}

dataset_to_output_key = {
    "MathInstruct": "output",
    "python_code_instructions": "output",
    "dolly": "response",
    "billsum": "summary",
    "commitpackft": "output",
    "NuminaMath-CoT": "solution",
    "PubMedQA": "final_decision",
    "MetaMathQA": "response",
    "evol_instruct": "response",
    "cnn_dailymail": "highlights",
    "xsum": "summary",
    "meetingbank": "summary",
}

column_map = {
    "MathInstruct": {"instruction": "instruction"},
    "python_code_instructions": {"instruction": "instruction", "input": "input"},
    "dolly": {"instruction": "instruction", "input": "context"},
    "billsum": {"dialogue": "text"},
    "commitpackft": {"instruction": "instruction", "input": "input"},
    "NuminaMath-CoT": {"question": "problem"},
    "PubMedQA": {"question": "question"},
    "MetaMathQA": {"question": "query"},
    "evol_instruct": {"instruction": "instruction"},
    "cnn_dailymail": {"dialogue": "article"},
    "xsum": {"dialogue": "document"},
    "meetingbank": {"dialogue": "transcript"},
}

def format_prompt(dataset_name, root_folder="data"):
    if dataset_name not in dataset_to_template:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    if os.path.exists(f"{root_folder}/{dataset_name}/{dataset_name}.json"):
        json_file = f"{root_folder}/{dataset_name}/{dataset_name}.json"
    else:
        json_file = _load_dataset_to_json(dataset_name, root_folder=root_folder)

    datas = []
    try:
        jdict = jload(json_file)
    except BaseException:
        with open(json_file, 'r') as f:
            lines = f.readlines()
        jdict = [json.loads(line.strip()) for line in lines]    
    print(jdict[0].keys())

    for example in jdict:
        text = dataset_to_template[dataset_name].format(example, column_map.get(dataset_name, {}))
        example['text'] = text + example[dataset_to_output_key[dataset_name]]
        datas.append(example)

    jdump(datas, f"{root_folder}/{dataset_name}/{dataset_name}.json")

    new_jdict = jload(f"{root_folder}/{dataset_name}/{dataset_name}.json")
    # print(new_jdict[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="PubMedQA"
    )
    args = parser.parse_args()
    format_prompt(args.dataset_name, root_folder)