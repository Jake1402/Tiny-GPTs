import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from argparse import RawTextHelpFormatter
import argparse

import numpy as np
from time import time
import json

from GPT import GPT, save_model, load_model
from Tokeniser import SimpleTokenizer

def args_builder():
    parser = argparse.ArgumentParser(
    description="""
    Training-Pytorch-SingleGPU.py is responsible for training your model
    on a single GPU or CPU machine. This module can be used for both
    pretraining and finetuning your model. We recommend RLHF to get optimal
    results after training, unfortunatly RLHF isn't included in the model. 
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
        help="The name of your vocab file."
    )
    parser.add_argument(
        "--training_set",
        type=str,
        required=True,
        help="The name of your training set (files must end in Feature/Label.npy DON'T INCLUDE)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=30,
        help="Number of epochs for training (Default 30)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=False,
        default=64,
        help="The batch size of your training data (Default 64)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate for the model (Default 1e-3)."
    )
    parser.add_argument(
        "--grad_norm",
        type=float,
        required=False,
        default=1.0,
        help="Gradient clipping to stabilise gradients (Default 1.0)."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    torch.cuda.empty_cache() 
    args = args_builder()
    MODEL_NAME = f"models/{args.model}"
    DIR_OF_VOCAB_ = f"vocabs/{args.vocab}.json"                           # YOUR VOCAB
    FEATURES_DIR_ = f"compiled-datasets/{args.training_set}-Features.npy" # YOUR FEATURES
    _LABELS__DIR_ = f"compiled-datasets/{args.training_set}-Labels.npy"   # YOUR LABELS
    global PADDING_ID, PADDING_IGNORE
    
    tokenizer = SimpleTokenizer()   
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True

    PADDING_ID = tokenizer.pad_id
    PADDING_IGNORE = -100

    SAVE_EVERY    = 5               # Save a model every 5 epochs
    EPOCHS        = args.epochs     # Number of epochs based on Args
    GRAD_CLIPS    = args.grad_norm  # The gradient clipping value
    LEARNING_RATE = args.lr         # Models learning rate
    BATCH         = args.batch      # The batch size (smaller == less vram use)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available else CPU.
    model = load_model(MODEL_NAME).to(device)                             # Loading the built model.
    print(f"Parameter count : {model.returnParams()}")                    # Printing the models parameter count.

    loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_IGNORE)  # Using Cross Entropy Loss with ingnore index
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)   # AdamW is being used as the optimiser

    np_features = np.load(FEATURES_DIR_, allow_pickle=True) # Loading features from directory.
    np_labels   = np.load(_LABELS__DIR_, allow_pickle=True) # Loading Lavels   from directory.
    print(np_features.shape)                                # Printing the shape of features.
    print(np_labels.shape)                                  # Printing the shape of labels.

    training_feat = []
    training_labl = []

    print("Loading dataset")                                        
    for item in zip(np_features, np_labels):                                # Iterating through pairs of dataset.
        training_feat.append(torch.as_tensor(np.astype(item[0], np.int64))) # Converting features to torch tensors
        training_labl.append(torch.as_tensor(np.astype(item[1], np.int64))) # Converting labels   to torch tensors
    ds = TensorDataset(                             # Using pytorchs TensorDataset.
        torch.as_tensor(np.array(training_feat)),   # Convert lists to numpy then to torch
        torch.as_tensor(np.array(training_labl)))   # This is due to pytorch not playing well with lists.
    dl = DataLoader(      # Pytorch dataloader to load X/Y values.
        dataset=ds,       # Passing the tensor dataset.
        batch_size=BATCH, # Passing batch size 
        shuffle=True,     # Shuffle improves training stability
        num_workers=16,   # Num workers (num threads for loading X/Y)
        pin_memory=True   # Pin memory speeds up transfer by locking memory
    )                     

    print("Beginning training")
    model.train(True)
    
    for epoch in range(1, EPOCHS + 1):
        init_time = time()
        loss_avg = 0
        nums_avg = 0
        for index, (features, labels) in enumerate(dl):
            nums_avg += features.shape[0]
            features = features.to(device)
            labels = labels.to(device)

            predictions = model.forward(features, train=True)
            loss = loss_fn(predictions.transpose(1, 2), labels)
            loss_avg += (loss*features.shape[0])

            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIPS)
            optim.step()

        print(f"Current Epoch - {epoch}, current avg loss - {loss_avg/nums_avg:.4f}, Time took - {time() - init_time}")    
        if epoch % SAVE_EVERY == 0:
            save_model(f"models/Pretrained-{epoch}", model)
    save_model(f"models/Pretrained-Finished", model)
    print("Training complete.")
