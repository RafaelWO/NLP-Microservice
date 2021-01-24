from transformers import AutoTokenizer, AutoModelForCausalLM

model_str = "gpt2"
AutoTokenizer.from_pretrained(model_str).save_pretrained(f"local_model/{model_str}/tokenizer")
AutoModelForCausalLM.from_pretrained(model_str).save_pretrained(f"local_model/{model_str}/model")
