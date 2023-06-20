# 1. Library imports
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse, Response
from web.config import Configuration
from web.utils import json_to_yaml, zip_folder, get_dataset
from preprocess import main as ddsp_preprocess
from train import main as ddsp_train
from export import main as ddsp_export
import os
from subprocess import Popen
import shutil

# 2. Create app and model objects
app = FastAPI()


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post('/pull_data')
def pull_data(data_id):
    # Get HuggingFace secret token
    HF_TOKEN = os.environ.get("HF_TOKEN")

    # Pull dataset
    get_dataset(data_id, HF_TOKEN)

    return Response(status_code=201)


@app.post('/train')
def train(config: Configuration):
    # Get JSON config
    config_instance = Configuration.parse_obj(config)

    # Convert JSON to YAML
    json_to_yaml(config_instance)

    # Preprocess data
    # ddsp_preprocess()

    # Open up asynchronous instance of tensorboard
    board = Popen(["tensorboard", "--logdir=" + os.getcwd() + "/models"])

    # Train model
    ddsp_train()

    # Close tensorboard
    board.terminate()

    return Response(status_code=201)


@app.post('/export')
def export(model_name):
    # Export model
    ddsp_export(model_name)

    # Comporess model as zip file and return via post request
    file_name = model_name + ".zip"
    zip_folder(model_name, os.path.join("models", file_name))

    # Remove original model directory
    shutil.rmtree(model_name)

    return FileResponse(os.path.join("models", file_name), filename=file_name)


@app.post('/clear_models')
def clear_models():

    shutil.rmtree('models')

    return Response(status_code=204)


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
