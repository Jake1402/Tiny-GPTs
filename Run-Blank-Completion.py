import torch
import json

from GPT import GPT, load_model
from dataset.simple_tokenizer import SimpleTokenizer, SUFFIXES, SPECIAL_SPACES


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    DIR_OF_VOCAB_ = "dataset/vocabs/wiki-vocab.json"
    MODEL_TO_USE_ = "./models/Finetuned-10"
    
    tokenizer = SimpleTokenizer()
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True
    
    with open(f"{MODEL_TO_USE_}.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    VOCAB_SIZE = 12110
    D_MODEL    = 512
    LAYERS     = 24
    MASKED     = 1
    NUM_HEADS  = 8
    MAX_TOKENS = 2048

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    badgerMLM = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        layers=LAYERS,
        masked=MASKED,
        num_heads=NUM_HEADS,
        max_tokens=MAX_TOKENS
    ).to("cuda:0")
    
    badgerMLM = load_model(MODEL_TO_USE_)
    
    print(f"Parameter count : {badgerMLM.returnParams()}")

    badgerMLM.load_state_dict(torch.load("./models/Finetuned-10.pt", weights_only=True))
    badgerMLM.train(False)
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
                encoded += tokenizer.encode(text, add_bos=True, add_eos=False, max_len=256)   # Encode it
            sequence_len += 1   
            output = badgerMLM.forward(torch.tensor(encoded).unsqueeze(0).to(device), temperature=0.8).cpu()
            encoded.append(output.squeeze(0).tolist()[-1])
            decoded = tokenizer.decode(output.squeeze(0).tolist(), skip_specials=False)
            if decoded.split()[-1] in (SUFFIXES+SPECIAL_SPACES):
                text += decoded.split()[-1]
            else:
                text += " " + decoded.split()[-1]
    print(text)