
import cs336_basics
import json


def get_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def main():
    config = get_config("train/train.json")
    print("config:", config)
    print("hello world")


if __name__ == "__main__":
    main()
