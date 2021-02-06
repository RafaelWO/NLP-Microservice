from flask import Flask, request
from flask_restx import Resource, Api, fields
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

app = Flask(__name__)
api = Api(app, version="0.1", title="AI Text-Generation")
ns = api.namespace("text-generation")

# Swagger defs
generation_input_def = api.model("Text Generation Input", {
    'text': fields.String(required=True,
                          description="Input prompt",
                          help="Text cannot be blank.",
                          example="Artificial Intelligence is a")
})


model_str = "gpt2"
print(f"Using model '{model_str}'")
print("Loading tokenizer...", end=' ')
tokenizer = AutoTokenizer.from_pretrained(f"local_model/{model_str}/tokenizer",
                                          config=AutoConfig.from_pretrained(f"local_model/{model_str}/model"))
print("Done")
print("Loading model...", end=' ')
model = AutoModelForCausalLM.from_pretrained(f"local_model/{model_str}/model")
print("Done")
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'CPU' if device == -1 else 'GPU'} as device.")
print("Building pipeline...", end=' ')
generate_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
print("Done")


@ns.route('/generate')
class Conversation(Resource):
    @ns.expect(generation_input_def)
    def post(self):
        input_text = request.json['text']
        out = generate_pipe(input_text, return_tensors=True, return_text=False)

        # Remove input from the generated text
        out_ids = out[0]['generated_token_ids']
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        generated_text = tokenizer.decode(out_ids[len(input_ids):])

        return {'input': input_text, 'generated': generated_text}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
