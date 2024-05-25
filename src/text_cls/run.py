import yaml
import os


if __name__ == "__main__":
    # region 1. Load YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    # endregion