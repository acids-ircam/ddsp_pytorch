# 1. Library imports
import uvicorn
from fastapi import FastAPI
from web.config import Configuration
from web.utils import json_to_yaml
import preprocess

# 2. Create app and model objects
app = FastAPI()


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/train')
def train(config: Configuration):
    config_instance = Configuration.parse_obj(config)

    config_yaml = json_to_yaml(config_instance)

    preprocess.main()
    # train.main()
    # export.main()

    return "Success!"


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
