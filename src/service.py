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
