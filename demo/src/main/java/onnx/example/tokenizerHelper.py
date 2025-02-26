
import json
from transformers import AutoTokenizer
import sys


model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenizeText(input_text):
    inputs = tokenizer(input_text, return_tensors="np")
    #print("tokenized inputs: ", inputs)

    #convert numpy arrays to lists
    inputs_list = {k:v.tolist() for k, v in inputs.items()}
    #print("converted input: ", inputs_list)

    return inputs_list


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello World" # get the argument passed
    tokens = tokenizeText(text)
    print(json.dumps(tokens))


