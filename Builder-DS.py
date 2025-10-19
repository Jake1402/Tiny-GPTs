from Tokeniser import SimpleTokenizer
from unidecode import unidecode
import numpy as np
import json
import pandas as pd
from random import random
import re
from argparse import RawTextHelpFormatter
import argparse

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def args_builder():
    parser = argparse.ArgumentParser(description="""ds_builder.py Will build your datasets as save them in npy file format.
Output format will be the tokenised data with features being the 
initial inputs and labels the output shifted right. Beginning/End
of sequence tags are included by default.
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="The name of the vocab file saved in vocabs"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Dataset csv file (NO INDEX)"
    )
    parser.add_argument(
        "--columns",
        type=int,
        required=True,
        help="Should be either 1 or 2."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="example",
        help="Header name for saved files (default 'Example')"
    )
    parser.add_argument(
        "--length",
        type=int,
        required=False,
        default=512,
        help="truncate examples to desired length (default 512)"
    )
    parser.add_argument(
        "--hide_rate",
        type=float,
        required=False,
        default=0.2,
        help="The chance of hiding tokens (default 0.2)"
    )
    parser.add_argument(
        "--roles",
        type=bool,
        required=False,
        default=False,
        help="Decides to add roles like OPERATOR/AGENT"
    )
    parser.add_argument(
        "--pad",
        type=bool,
        required=False,
        default=True,
        help="Decides pad sequences to max length default true"
    )
    parser.add_argument(
        "--operation",
        type=int,
        required=False,
        default=2,
        help="""Decides which dataset to generate:
        0 : Next token prediction
        1 : Fill the empty spaces
        2 : Generates both.
        """
    )
    parser.add_argument(
        "--tags",
        type=int,
        required=False,
        default=0,
        help="""Decides if <eos>/<bos> tags are added:
        0 : None
        1 : <bos> (added to front of user prompt)
        2 : <eos> (added to end of AI text)
        3 : <bos> and <eos> in 1/2 positions.
        Default is None are added
        """
    )
    args = parser.parse_args()
    return args

def text_chunking(text, target_tokens=400):
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_count + sentence_words > target_tokens and current_chunk:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = [sentence]
            current_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_count += sentence_words

    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    return chunks

def next_token_prediction(text_list, tokenizer, max_len=512, name_header="", inc_tags=(1,1), pad=True):
    EOS, BOS = inc_tags
    FEATURE_LIST = []
    LABELS_LIST  = []
    for text in text_list:
        complete = tokenizer.encode(text, add_bos=BOS, add_eos=EOS, max_len=max_len)
        
        feature = np.asarray(complete[:-1], dtype=np.int64)
        label   = np.asarray(complete[1:], dtype=np.int64)
        if pad:
            feature = np.pad(feature, pad_width=[0,max_len - len(feature)], constant_values=tokenizer.pad_id)
            label = np.pad(label,     pad_width=[0,max_len - len(label)],   constant_values=-100)

        FEATURE_LIST.append(feature)
        LABELS_LIST.append(label)
    feature_save = np.asarray(FEATURE_LIST, dtype="O")
    np.save(os.path.join(__location__, f"compiled-datasets/{name_header}-Text-Generation-Features.npy"), feature_save)
    
    label_save = np.asarray(LABELS_LIST, dtype="O")
    np.save(os.path.join(__location__, f"compiled-datasets/{name_header}-Text-Generation-Labels.npy"), label_save)
    
def fill_empty(text_list, tokenizer, padding_id, missing_chance=0.1, max_len=512, name_header=", ", pad=True):
    FEATURE_LIST = []
    LABELS_LIST  = []
    for text in text_list:
        full_token = unidecode(text)
        full_token = text_chunking(full_token, int(max_len*0.5))
        for token_chunks in full_token:
            masked   = tokenizer.encode(token_chunks, add_bos=1, add_eos=1, max_len=max_len//2)
            unmasked = tokenizer.encode(token_chunks, add_bos=0, add_eos=1, max_len=max_len//2)
            
            for index in range(1, len(masked)-2):
                if random() < missing_chance:
                    masked[index] = padding_id
            
            predictor = masked + unmasked
            
            masked = np.asarray(predictor[:-1], dtype=np.int64)
            unmasked   = np.asarray(predictor[1:], dtype=np.int64)
            
            if pad:
                feature = np.pad(masked, pad_width=[0,max_len - len(masked)], constant_values=tokenizer.pad_id)
                label = np.pad(unmasked, pad_width=[0,max_len - len(unmasked)], constant_values=-100)
            
            FEATURE_LIST.append(feature)
            LABELS_LIST.append(label)

    feature_save = np.asarray(FEATURE_LIST, dtype="O")
    label_save = np.asarray(LABELS_LIST, dtype="O")
    np.save(os.path.join(__location__, f"compiled-datasets/{name_header}-Text-Missing-Features.npy"), feature_save)
    np.save(os.path.join(__location__, f"compiled-datasets/{name_header}-Text-Missing-Labels.npy"), label_save)    

if __name__ == "__main__":
    args = args_builder()
    VOCAB_NAME_ = args.vocab        # Vocab file name excluding .csvjson
    CSV_DATASET = args.csv          # csv dataset name excluding .csv
    COLUMN_NUMS = args.columns      # number of columns
    NPY_DS_NAME = args.name         # Header name of npy file
    MAX_LENGTH_ = args.length       # Truncation length of tokeniser
    _HIDE_RATE_ = args.hide_rate    # Hidden rate of tokens
    _ADD_ROLES_ = args.roles        # Add roles to text useful for chatbot
    ADDING_PAD_ = args.pad
    _OPERATION_ = args.operation    # Which operation to perform
    ADDING_TAGS = args.tags         # Adds <bos>/<eos> tags

    if ADDING_TAGS == 1:    # If option one
        INC_TAGS = (1, 0)   # Add only <bos>
    elif ADDING_TAGS == 2:  # If option two
        INC_TAGS = (0, 1)   # Add only <eos>
    elif ADDING_TAGS == 3:  # If option three
        INC_TAGS = (1, 1)   # Add both <bos>/<eos>
    else:                   # If anything else 
        INC_TAGS = (0, 0)   # Default to no tags

    # Regex was used to scrub HTML tags from datasets. Source to the regex string is below.
    # https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    DIR_OF_VOCAB_ = os.path.join(__location__, f"vocabs/{VOCAB_NAME_}.json")          # Address of new vocab
    _DIR_OF_CSV_ = os.path.join(__location__, f"csv-datasets/{CSV_DATASET}.csv")   # File holding our conversations
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')   
    
    tokenizer = SimpleTokenizer()
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f: 
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True

    df = pd.read_csv(_DIR_OF_CSV_)
    print(df.head(5))
    text_list = []
    print("Loading dataset to list")
    for text in df.iterrows():
        if COLUMN_NUMS == 2:
            col_text_1 = re.sub(CLEANR, '', unidecode(str(text[1][0])))
            col_text_2 = re.sub(CLEANR, '', unidecode(str(text[1][1])))
        else:
            col_text_1 = re.sub(CLEANR, '', unidecode(str(text[1][0])))
            col_text_2 = " " 
        if _ADD_ROLES_:
            text_list.append(
                "OPERATOR: " + col_text_1 + " AGENT: " + col_text_2
            )
            continue
        text_list.append(
            col_text_1 + " " + col_text_2
        )       

    print("Processing dataset")
    if _OPERATION_ == 0:
        next_token_prediction(text_list, tokenizer=tokenizer, max_len=MAX_LENGTH_, name_header=NPY_DS_NAME, inc_tags=INC_TAGS, pad=ADDING_PAD_)
    elif _OPERATION_ == 1:
        fill_empty(text_list, tokenizer=tokenizer, missing_chance=_HIDE_RATE_, padding_id=tokenizer.unk_id, max_len=MAX_LENGTH_, name_header=NPY_DS_NAME, pad=ADDING_PAD_)
    elif _OPERATION_ == -1:
        print("Debugging works and files load")
    else:
        fill_empty(text_list, tokenizer=tokenizer, missing_chance=_HIDE_RATE_, padding_id=tokenizer.unk_id, max_len=MAX_LENGTH_, name_header=NPY_DS_NAME, pad=ADDING_PAD_)
        next_token_prediction(text_list, tokenizer=tokenizer, max_len=MAX_LENGTH_, name_header=NPY_DS_NAME, inc_tags=INC_TAGS, pad=ADDING_PAD_)
    print("Finished")