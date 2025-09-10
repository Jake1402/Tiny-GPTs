import re
import json
import pandas as pd
from unidecode import unidecode

from simple_tokenizer import SimpleTokenizer
from eda import next_token_prediction, fill_empty

if __name__ == "__main__":
    
    DIR_OF_VOCAB_ = "./dataset/vocabs/wiki-vocab.json"          # Address of new vocab
    CONVERSATIONS = "./dataset/conversations.csv"   # File holding our conversations
    MOVIE_REVIEWS = "./dataset/movie-reviews.csv"   # File holding movie reviews
    MINI_WIKI_TXT = "./dataset/mini-wiki-text.csv"
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')   
    
    tokenizer = SimpleTokenizer()
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True

    df = pd.read_parquet("hf://datasets/rahular/simple-wikipedia/data/train-00000-of-00001-090b52ccb189d47a.parquet")
    print(df.head())
    #df.to_csv("./dataset/mini-wiki-text.csv", index=False)
    
    text_list = []
    print("Loading dataset to list")
    for text in df.iterrows():
        if 15 < len(text[1][0].split()) < 512:
            text_list.append(re.sub(CLEANR, '', unidecode(text[1][0])))

    print(f"Size of mini wiki text is - {len(text_list)}")
    print("Processing dataset")
    #fill_empty(text_list, missing_chance=0.2, padding_id=tokenizer.unk_id)
    next_token_prediction(text_list, max_len=1024, name_header="wiki", tokenizer=tokenizer, inc_tags=(1,1))
    print("Finished")