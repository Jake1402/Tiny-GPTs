import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy

import json
import numpy as np

from GPT import GPT, save_model
from custom_datasets import TextDataset
from dataset.simple_tokenizer import SimpleTokenizer

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
        optimizer = torch.optim.Adam(self.gpt_model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False
    DIR_OF_VOCAB_ = "dataset/vocabs/small-vocab.json"
    FEATURES_DIR_ = "dataset/compiled-datasets/Text-Generation-Features.npy"
    _LABELS__DIR_ = "dataset/compiled-datasets/Text-Generation-Labels.npy"
    global PADDING_ID, PADDING_IGNORE
    
    tokenizer = SimpleTokenizer()
    with open(DIR_OF_VOCAB_, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {int(i): w for w, i in vocab.items()}
    tokenizer.fitted = True
    
    EPOCHS = 10
    GRAD_CLIPS = 1
    PRINT_EVERY = 64
    SAVE_EVERY = 1
    NUM_GPUS = 4
    BATCH = 96
    
    VOCAB_SIZE = 7824
    D_MODEL    = 128
    LAYERS     = 24
    MASKED     = True
    NUM_HEADS  = 4
    MAX_TOKENS = 512
    PADDING_ID = tokenizer.pad_id
    PADDING_IGNORE = -100
    
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        layers=LAYERS,
        masked=MASKED,
        num_heads=NUM_HEADS,
        max_tokens=MAX_TOKENS,
    )
    
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
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    L_GPT = LightningGPTWrapper(model, IGNORE_INDEX=PADDING_IGNORE)
    ddp = DDPStrategy(process_group_backend="gloo")
    trainer = Trainer(
        default_root_dir="models/", 
        max_epochs=30, 
        accelerator="gpu", 
        devices=NUM_GPUS, 
        strategy=ddp
    )
    trainer.fit(model=L_GPT, train_dataloaders=dl)
    save_model("Finished-Model", L_GPT)
