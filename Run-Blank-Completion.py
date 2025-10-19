import torch
import json

from GPT import GPT, load_model
from Tokeniser import SimpleTokenizer, SPECIAL_SPACES

from argparse import RawTextHelpFormatter
import argparse


def args_builder():
    parser = argparse.ArgumentParser(
    description="""
    Run-Blank-Completion.py is an interface to allow the user to interact with
    there model. It requires two arguments `model` and `vocab`.
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of your generated model."
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="The name of your models vocab."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = args_builder()
    DIR_OF_VOCAB_ = f"vocabs/{args.vocab}.json"
    MODEL_TO_USE_ = f"./models/{args.model}"
    
    tokenizer = SimpleTokenizer()
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True
    with open(f"{MODEL_TO_USE_}.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_TO_USE_).to(device)
    MAX_TOKENS = model.max_tokens
    model.eval()
    
    print(f"Parameter count : {model.returnParams()}")

    sequence_len = 0
    text = "STARTING"
    encoded = []
    with torch.no_grad():
        while (sequence_len < MAX_TOKENS): 
            if (tokenizer.eos_token in text.split()[-1]) or sequence_len == 0:
                print(text)         # Printing the text to screen.
                encoded=[]          # Clearing the encoded list.        
                sequence_len = 0    # Resetting sequence length as this model only works on current sentences.
                text = input(f"Enter prompt - ")    # Get the user input and add to old data.
                if "<EXIT>" in text:
                    break
                encoded += tokenizer.encode(text, add_bos=True, add_eos=True, max_len=MAX_TOKENS)   # Encode it
            sequence_len += 1   
            output = model.forward(torch.tensor(encoded).unsqueeze(0).to(device), temperature=0.6, top_k=10).cpu()
            encoded.append(output.squeeze(0).tolist()[-1])
            decoded = tokenizer.decode(output.squeeze(0).tolist(), skip_specials=True)
            if decoded.split()[-1] in (SPECIAL_SPACES):
                text += decoded.split()[-1]
            else:
                text += " " + decoded.split()[-1]
    print(text)