from flask import Flask, request
from flask_restx import Resource, Api, fields, namespace
from transformers import Conversation, pipeline, AutoTokenizer, AutoModelForCausalLM

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
tokenizer = AutoTokenizer.from_pretrained(f"local_model/{model_str}/tokenizer")
model = AutoModelForCausalLM.from_pretrained(f"local_model/{model_str}/model")
conv_pipe = pipeline("conversational", model=model, tokenizer=tokenizer)


@ns.route('/conversation')
class Conversation(Resource):
    @ns.expect(conversation_input_def)
    def post(self):
        input_text = request.json['text']
        # conversation = Conversation(input_text)
        # out = conv_pipe(conversation)

        return {'entered': input_text}


if __name__ == '__main__':
    app.run(debug=True)
