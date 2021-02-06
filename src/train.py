import torch
from transformers import pipeline

model_str = "gpt2"
print(f"Using model '{model_str}'")
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'CPU' if device == -1 else 'GPU'} as device.")
print("Building pipeline...", end=' ')
generate_pipe = pipeline("text-generation", model=model_str, tokenizer=model_str, device=device)
print("Done")


if __name__ == '__main__':
    # Call this file with: python train.py "This is the text I want to be continued"
    from codecs import decode
    import sys

    input_text = decode(sys.argv[1], 'unicode_escape')
    out = generate_pipe(input_text)
    print(out[0]['generated_text'])
