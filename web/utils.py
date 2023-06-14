import yaml
import json
from web.config import Configuration


def json_to_yaml(config: Configuration):
    # Convert JSON payload to yaml
    config_json = config.json()
    config_json_dict = json.loads(config_json)
    config_yaml = open("config.yaml", "w")
    yaml.dump(config_json_dict, config_yaml)
    config_yaml.close()
    return config_yaml
