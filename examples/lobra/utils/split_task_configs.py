import argparse
from trainer import TrainerConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/trainer_config.json")
    parser.add_argument("--split_num", type=int, default=1)
    args = parser.parse_args()
    trainer_config = TrainerConfig(args.config_path)
    trainer_config.split_and_dump_task_configs(args.split_num)