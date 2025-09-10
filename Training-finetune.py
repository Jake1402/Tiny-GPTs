import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
from time import time
import json

import GPT
from GPT import load_model
from custom_datasets import TextDataset
from dataset.simple_tokenizer import SimpleTokenizer

def collate_fn(batch):
    features, labels = zip(*batch)  # unzip list of tuples

    # Check if inputs are 1D or 2D
    if features[0].dim() == 1:
        padded_features = pad_sequence(features, batch_first=True, padding_value=PADDING_ID)
        padded_labels   = pad_sequence(labels,   batch_first=True, padding_value=PADDING_IGNORE)  # -100 ignored by CrossEntropy
    else:
        # Find max lengths in BOTH dimensions
        max_f0 = max(f.size(0) for f in features)
        max_f1 = max(f.size(1) for f in features)
        max_l0 = max(l.size(0) for l in labels)
        max_l1 = max(l.size(1) for l in labels)

        # Pad manually since pad_sequence only pads 1D/variable-length batches
        padded_features = torch.stack([
            torch.nn.functional.pad(f, (0, max_f1 - f.size(1), 0, max_f0 - f.size(0)), value=PADDING_ID)
            for f in features
        ])
        padded_labels = torch.stack([
            torch.nn.functional.pad(l, (0, max_l1 - l.size(1), 0, max_l0 - l.size(0)), value=PADDING_IGNORE)
            for l in labels
        ])

    return padded_features, padded_labels

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    DIR_OF_VOCAB_ = "dataset/vocabs/wiki-vocab.json"
    FEATURES_DIR_ = "dataset/compiled-datasets/instruct-Text-Generation-Features.npy"
    _LABELS__DIR_ = "dataset/compiled-datasets/instruct-Text-Generation-Labels.npy"
    global PADDING_ID, PADDING_IGNORE
    
    tokenizer = SimpleTokenizer()
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True
    
    VOCAB_SIZE = 12110
    D_MODEL    = 512
    LAYERS     = 24
    NUM_HEADS  = 8
    MAX_TOKENS = 2048
    PADDING_ID = tokenizer.pad_id
    PADDING_IGNORE = -100

    EPOCHS = 10
    GRAD_CLIPS = 1
    PRINT_EVERY = 64
    SAVE_EVERY = 1
    BATCH = 70
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #devices = [i for i in range(torch.cuda.device_count())]
    #device="cpu"           # For debugging cuda problems (usual come from dim errors)
    badgerMLM = GPT.GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        layers=LAYERS,
        num_heads=NUM_HEADS,
        max_tokens=MAX_TOKENS
    ).to(device)
    badgerMLM.load_state_dict(torch.load("./models/Pretrained-Model.pt", weights_only=True))
    print(f"Parameter count : {badgerMLM.returnParams()}")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_IGNORE)           # Using Cross Entropy Loss
    optim = torch.optim.AdamW(badgerMLM.parameters(), lr=0.00001) # Adam is being used as the optimiser

    np_features = np.load(FEATURES_DIR_, allow_pickle=True)
    np_labels   = np.load(_LABELS__DIR_, allow_pickle=True)
    print(np_features.shape)
    print(np_labels.shape)

    training_feat = []
    training_labl = []

    print("Loading dataset")
    for item in zip(np_features, np_labels):
        training_feat.append(torch.as_tensor(np.astype(item[0], np.int64)))
        training_labl.append(torch.as_tensor(np.astype(item[1], np.int64)))
    ds = TextDataset(training_feat, training_labl)
    dl = DataLoader(
        dataset=ds,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Beginning training")
    global_step = 0
    badgerMLM.train(True)
    
    for epoch in range(1, EPOCHS + 1):
        init_time = time()
        for index, (features, labels) in enumerate(dl):
            features = features.to(device)
            labels = labels.to(device)
            predictions = badgerMLM.forward(features, train=True)
            loss = loss_fn(predictions.transpose(1, 2), labels)
            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(badgerMLM.parameters(), max_norm=GRAD_CLIPS)
            del features, labels
            optim.step()
            if index % PRINT_EVERY == 0:
                print(f"Current batch number - {index}, current loss - {loss}, epoch number - {epoch}")
                
        print(f"Time took - {time() - init_time}")
        if epoch % SAVE_EVERY == 0:
            badgerMLM.save_model(f"Finetuned-{epoch}")
    print("Training complete.")
