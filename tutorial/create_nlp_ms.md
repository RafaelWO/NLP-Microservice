# NLP-Microservice

## 1. Environment
- Create conda environment: `conda create -n nlp-ms python=3.8`
- Dependencies:
  - API Stuff:
    - [Flask](https://flask.palletsprojects.com/en/1.1.x/): `pip install Flask`
    - [Flask-RESTX](https://flask-restx.readthedocs.io/en/latest/): `pip install flask-restx`
  - [🤗Transformers](https://github.com/huggingface/transformers) is used as NLP library: `pip install transformers`
  - [TensorFlow 2.0]() or [PyTorch]()
    - I used Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation instructions) 
  
## 2. Simple Flask-RESTX App
```Python
# FILE: service.py

from flask import Flask
from flask_restx import Resource, Api

app = Flask(__name__)
api = Api(app)


@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


if __name__ == '__main__':
    app.run()
```

![simple-flask-app](./images/NLP-ms-1.gif)

## 3. Add NLP Pipeline
```Python
# FILE: service.py

from flask import Flask, request
from flask_restx import Resource, Api, fields
from transformers import pipeline
from transformers import Conversation as ConversationHelper

app = Flask(__name__)
api = Api(app, version="0.1", title="AI Conversation")
ns = api.namespace("conversation")

# Swagger defs
conversation_input_def = api.model("Conversation Input", {
    'text': fields.String(required=True,
                          description="Start of conversation",
                          help="Text cannot be blank.",
                          example="What is the meaning of life?")
})


conv_pipe = pipeline("conversational")      # Use default model of pipeline


@ns.route('/conversation')
class Conversation(Resource):
    @ns.expect(conversation_input_def)
    def post(self):
        input_text = request.json['text']
        conversation = ConversationHelper(input_text)
        out = conv_pipe(conversation)

        return {'conversation': repr(out)}


if __name__ == '__main__':
    app.run()
```

![simple-nlp-pipeline](./images/NLP-ms-2.gif)

## 4. Pre-download Model in separate Script
This is very useful for running the API in a Docker container (see TODO).
```Python
# FILE: download.py

from transformers import AutoTokenizer, AutoModelForCausalLM

model_str = "microsoft/DialoGPT-medium"
AutoTokenizer.from_pretrained(model_str).save_pretrained(f"local_model/{model_str}/tokenizer")
AutoModelForCausalLM.from_pretrained(model_str).save_pretrained(f"local_model/{model_str}/model")
```

Now the saved model can be used:
```Python
# FILE: service.py

from flask import Flask, request
from flask_restx import Resource, Api, fields
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Conversation as ConversationHelper

app = Flask(__name__)
api = Api(app, version="0.1", title="AI Conversation")
ns = api.namespace("conversation")

# Swagger defs
conversation_input_def = api.model("Conversation Input", {
    'text': fields.String(required=True,
                          description="Start of conversation",
                          help="Text cannot be blank.",
                          example="What is the meaning of life?")
})


model_str = "microsoft/DialoGPT-medium"
print(f"Using model '{model_str}'")
print("Loading tokenizer...", end=' ')
tokenizer = AutoTokenizer.from_pretrained(f"local_model/{model_str}/tokenizer",
                                          config=AutoConfig.from_pretrained(f"local_model/{model_str}/model"))
print("Done")
print("Loading model...", end=' ')
model = AutoModelForCausalLM.from_pretrained(f"local_model/{model_str}/model")
print("Done")
print("Building pipeline...", end=' ')
conv_pipe = pipeline("conversational", model=model, tokenizer=tokenizer)
print("Done")


@ns.route('/conversation')
class Conversation(Resource):
    @ns.expect(conversation_input_def)
    def post(self):
        input_text = request.json['text']
        conversation = ConversationHelper(input_text)
        out = conv_pipe(conversation)

        return {'conversation': repr(out)}


if __name__ == '__main__':
    app.run()
```