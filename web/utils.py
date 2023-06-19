import yaml
import json
import os
import requests
import soundfile as sf
from datasets import load_dataset
from web.config import Configuration


def json_to_yaml(config: Configuration):
    # Convert JSON payload to yaml
    config_json = config.json()
    config_json_dict = json.loads(config_json)
    config_yaml = open("config.yaml", "w")
    yaml.dump(config_json_dict, config_yaml)
    config_yaml.close()


def get_dataset(dataset_id, api_token):
    # Load the dataset with authentication
    dataset = load_dataset(dataset_id, use_auth_token=api_token)
    dataset.save_to_disk('data')
    audio_data = dataset['train']  # Access the 'train' split of the dataset

    # Iterate over the dataset and save the .wav files
    for i, example in enumerate(audio_data):
        data = example['audio']  # Assuming the audio column is named 'audio'
        sr = data['sampling_rate']
        x = data['array']

        file_path = f"data/audio_{i}.wav"  # Name the file with a unique identifier
        sf.write(file_path, x, samplerate=sr)  # Save the audio data

        print(f"Saved audio file: {file_path}")
