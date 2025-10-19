import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy

from argparse import RawTextHelpFormatter
import argparse

import json
import numpy as np

from GPT import GPT, save_model, load_model
from Tokeniser import SimpleTokenizer

def args_builder():
    parser = argparse.ArgumentParser(
    description="""
    Training-Lightning-MultiGPU.py is responsible for training your models by
    utilising multiple GPUs. This is done by utilising PyTorch Lightning for safely
    training on multiple GPUs.
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
        "--gpus",
        type=int,
        required=True,
        default=1,
        help="Number of GPUs to use during training (Default 1)."
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


class LightningGPTWrapper(L.LightningModule):
    def __init__(self, gpt_model : GPT, IGNORE_INDEX : int):
        super().__init__()
        self.gpt_model = gpt_model
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.gpt_model.forward(x, train=True)
        loss = self.loss(predictions.transpose(1, 2), y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gpt_model.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [lr_scheduler]

if __name__ == "__main__":

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
    
    SAVE_EVERY    = 5               # Save a model every 5 epochs
    NUM_GPUS      = args.gpus       # The number of GPUs to use
    EPOCHS        = args.epochs     # Number of epochs based on Args
    GRAD_CLIPS    = args.grad_norm  # The gradient clipping value
    LEARNING_RATE = args.lr         # Models learning rate
    BATCH         = args.batch      # The batch size (smaller == less vram use)

    PADDING_ID = tokenizer.pad_id
    PADDING_IGNORE = -100
    
    model = load_model(MODEL_NAME)
    
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
    
    ds = TensorDataset(torch.as_tensor(np.array(training_feat)), torch.as_tensor(np.array(training_labl)))
    dl = DataLoader(
        dataset=ds,
        batch_size=BATCH,
        num_workers=16,
        pin_memory=True
    )
    
    L_GPT = LightningGPTWrapper(model, IGNORE_INDEX=PADDING_IGNORE)
    ddp = DDPStrategy(process_group_backend="gloo")
    trainer = Trainer(
        default_root_dir="models/", 
        max_epochs=EPOCHS, 
        accelerator="gpu", 
        devices=NUM_GPUS, 
        strategy=ddp,
        gradient_clip_val=1,
        
    )
    trainer.fit(model=L_GPT, train_dataloaders=dl)
    save_model("models/Finished-Model", model)