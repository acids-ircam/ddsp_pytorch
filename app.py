# 1. Library imports
import uvicorn
from fastapi import FastAPI
from web.config import Configuration
from web.utils import json_to_yaml, get_dataset
from preprocess import main as preprocess
# from train import main as train
# from export import main as export
import os

# 2. Create app and model objects
app = FastAPI()


@app.post('/pull_data')
def pull_data(data_id):
    HF_TOKEN = os.environ.get("HF_TOKEN")
    get_dataset(data_id, HF_TOKEN)

    return "Success!"


@app.post('/train')
def train(config: Configuration):
    config_instance = Configuration.parse_obj(config)

    config_yaml = json_to_yaml(config_instance)

    preprocess()
    # train()

    return "Success!"


@app.post('/export')
def export(file_name):
    # export()

    return "Success!"


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
