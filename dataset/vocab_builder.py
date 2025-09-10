import json
from unidecode import unidecode
from simple_tokenizer import SimpleTokenizer
import re
import pandas as pd

#from argparse import RawTextHelpFormatter
import argparse

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def args_builder():
    parser = argparse.ArgumentParser(description="""
        vocab_builder.py will build your vocab for you and store it in a json file. It will only add words if they appear more
        than F number of times. This way random phrases or misspellings are more likely to be avoided. It will build its
        dataset from a csv dataset file. I recommend using the mini-wiki-text.csv file.
    """)
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of new vocab"
    )
    parser.add_argument(
        "--f",
        type=int,
        required=False,
        default=5,
        help="""Add words if appear more than F.
        Default is 5.
        """
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Read from csv dataset"
    )
    parser.add_argument(
        "--skip",
        type=int,
        required=False,
        default=10,
        help="""number of rows to skip in dataset.
        Prevents large dataset explosion while
        keeping the vocab diverse.
        Default is 10.
        """
    )
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    

    args = args_builder()
    VOCAB_NAME_ = os.path.join(__location__, f"vocabs/{args.name}.json")
    M_FREQUNCY_ = args.f
    CSV_DATASET = os.path.join(__location__, f"csv-datasets/{args.csv}.csv")

    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    f = open(VOCAB_NAME_, "w")
    f.write('{ "a" : 0 }')
    f.close()
    
    tokenizer = SimpleTokenizer()
    with open(VOCAB_NAME_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True

    df = pd.read_csv(CSV_DATASET) # Reading the text conversations as a pandas dataframe
    print(df.head(5))               # Print first five rows to confirm shape/column names
    
    total = [] #
    line_count = 0                      # Keep track of line number
    for line in df.iterrows():          # Iterate through each row of the dataset.
        line_count += 1                 # Incriment line number
        if line_count % 10 != 0:        # Skip 9/10 rows (add variation to vocab)
            continue                    # Skipping row
        prompt = line[1].iloc[0]                # Read column one which is 'question' and store in prompt
        text = unidecode(" " + prompt)          # Remove special characters.
        total.append(re.sub(CLEANR, '', text))  # Add the strings together with spaces inbetween.

    vocab_list = tokenizer.build_vocab(total, min_freq=M_FREQUNCY_)   # Convert this list to a dictionary
    print(f"Total length of vocab - {len(vocab_list)}")     # Print the length of the vocabulary

    f = open(VOCAB_NAME_, "w")                        # Create or open existing vocab file.
    f.write(json.dumps(vocab_list, indent=4))       # Convert dictionary to json string and write to new file
    f.close()                                       # Close the file.