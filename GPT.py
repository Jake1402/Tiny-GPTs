import torch.nn as nn
import torch
import numpy as np
import json

import torch.nn.functional as F

"""
I did not program the positional encoding class. Instead the 
tutorial from StatQuest was used and will be linked at the end 
of this comment. I will say it was fantastic and was brilliant
at helping me understand Positional Encoding.
    https://github.com/StatQuest/decoder_transformer_from_scratch
"""
class PositionEncoding(nn.Module):
    def __init__(self, d_model = 300, max_len=2048):
        super(PositionEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index/d_model)

        pe[:, 0::2] = torch.sin(position*div_term)
        """
        Due to how the function layed out, if we don't include this we have a dimension
        mismatch. To prevent this an if conditional is used and we can use odd numbered
        embedding sizes. Credit goes to:
            https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986/2
        """
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)

    def forward(self, word_embedding, training = False):
        if not training:
            return word_embedding + self.pe[:word_embedding.size(0), :]     # Adds word embedding based on batch size
        return word_embedding + self.pe[:word_embedding.size(-2), :].unsqueeze(0).repeat(word_embedding.size(0), 1, 1)

class WordEmbedder(nn.Module):
    def __init__(self, vocab_size=30522, d_model=512):
        super(WordEmbedder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, word_ids):
        return self.embedding_layer(word_ids)

"""
Block class was written by me following the documentation on pytorch
and following the general idea behind Block only models and the
"Attention is all you need paper"
"""
class Block(nn.Module):
    def __init__(self, d_model = 512, num_heads = 8, sequence_len = 2048, masked=True):
        super(Block, self).__init__()

        self.d_model      = d_model                 # D model size is the embedding size of the model
        self.num_heads    = num_heads               # Number of self attention heads we want to use
        self.sequence_len = sequence_len            # The maximum length of our sequences
        self.masked       = masked
        self.dff          = self.d_model*4          # The hidden layer size in the FCN layerd

        self.wQ = nn.Linear(in_features=d_model, out_features=d_model, bias=False)                          # The Queries matrix
        self.wK = nn.Linear(in_features=d_model, out_features=d_model, bias=False)                          # The Keys matrix
        self.wV = nn.Linear(in_features=d_model, out_features=d_model, bias=False)                          # The Value matrix
        self.Attention  = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)   # The multi head attention 

        self.attentionMask = None       # The attention mask is defined a None type here as its size is dynamic
        self.norms = nn.RMSNorm(d_model)

        self.FCN = nn.Sequential(                                       # Fully conncted layer after attention
            nn.Linear(in_features=d_model, out_features=self.dff),      # Input size is d_model folowed by out_features
            nn.Dropout(0.4),                                            # Add dropout to prevent overfitting certain neurones
            nn.GELU(),                                                  # Use ReLU activation as activation
            nn.Linear(in_features=self.dff, out_features=d_model),      # Another linear layers inputs is DFF out is m_mod
            nn.Dropout(0.4),                                            # Add dropout to prevent overfitting certain neurones
            nn.GELU()                                                   # Activation of ReLU again.
        )

    def forward(self, inputs):

        Queries = self.wQ(inputs)       # Calculate Queries
        Keys    = self.wK(inputs)       # Calculate Keys
        Values  = self.wV(inputs)       # Calculate Values

        self.attentionMask = torch.triu(
            torch.ones(inputs.size(1), inputs.size(1))*-torch.inf, diagonal=1
        ).to(Queries.device)          
        attention, _ = self.Attention.forward(Queries, Keys, Values, attn_mask=self.attentionMask)  # Calculate attention (using forward for mask)
        if self.masked:
            attention = torch.where(                                                                                    # Masking the attention using the "where" function
                torch.isneginf(attention), torch.tensor(0.0, dtype=attention.dtype, device=attention.device),           # This will find all neginf values and replace 
                attention                                                                                               # them with 0. Masking our attention!
            )
        attention = self.norms(attention + inputs)     # Add the attention to the inputs and normalise
        FFNOutput = self.FCN(attention)         # Calculate FFN layer
        return self.norms(FFNOutput + attention)                        # Return the FFN layer output


"""
    A GPT Arhcitecture as discussed in the paper:
        - Language models are few shot learners
    The process for this implementation allows for easy
    modification and customisability of the GPT architecture.
"""
class GPT(nn.Module):
    def __init__(self,vocab_size : int, d_model : int, layers : int, masked : bool, num_heads : int, max_tokens : int):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size    # Vocab size of our model
        self.d_model    = d_model       # Embedding features size
        self.layers     = layers        # Number of decoder layers
        self.masked     = masked        # Masked or unmasked (decoder v encoder)
        self.num_heads  = num_heads     # Number of self attention heads
        self.max_tokens = max_tokens    # Maximum number of allowed tokens

        if d_model % num_heads != 0:    # Check d_model num_head compatibility
            raise ValueError(f"d_model must be divisible by the number of heads.")

        self.embedder = WordEmbedder(vocab_size=self.vocab_size, d_model=self.d_model)
        self.pe = PositionEncoding(d_model=self.d_model, max_len=self.max_tokens)   # Setup a positional encoding object

        self.blocks = nn.ModuleList()               # Create a ModuleList of transformer blocks
        for _ in range(layers):                     # For loop for number of layers
            self.blocks.append(                     # Append blocks to the blocks list
                Block
                (
                    d_model=self.d_model,           
                    num_heads=self.num_heads, 
                    masked=self.masked,
                    sequence_len=self.max_tokens,
                )
            )

        self.projector = nn.Sequential(                               # Final layer in the transformer
            nn.Linear(in_features=d_model, out_features=vocab_size),  # Language modelling head.
        )

    def forward(self, inputs, train=False, temperature=1):
        """
            Inputs is an ID tensor. It should be a vector containing
            integer values. These values represent potential 
        """
        embeddings = self.embedder.forward(inputs)
        PE_inputs = self.pe.forward(embeddings, training=train)

        block_output = None
        for layer in range(self.layers):
            if layer == 0:
                block_output = self.blocks[layer].forward(inputs=PE_inputs)
                continue
            block_output = self.blocks[layer].forward(inputs=block_output)

        one_hot_output = self.projector(block_output)

        if train:
            return one_hot_output
        probability = torch.distributions.Categorical(probs=torch.nn.functional.softmax(one_hot_output/temperature, dim=-1))
        return probability.sample()

    def returnParams(self):

        #
        # Returns the total number of parameters
        # in the model. Very handy for comparisons
        #

        return sum(param.numel() for param in self.parameters() if param.requires_grad)
    
def save_model(name, model):
    model_dict = {
        "vocab" : model.vocab_size,
        "d_model" : model.d_model,
        "layers" : model.layers,
        "masked" : model.masked,
        "heads" : model.num_heads,
        "max_tokens" : model.max_tokens
    }
    f = open(f"{name}.json", "w")                     
    f.write(json.dumps(model_dict, indent=4))
    f.close()     
    torch.save(model.state_dict(), f"{name}.pt") 
        
def load_model(name):
    
    with open(f"{name}.json", "r", encoding="utf-8") as f:
        model_params = json.load(f)
    model = GPT(
        vocab_size=model_params["vocab"],
        d_model=model_params["d_model"],
        layers=model_params["layers"],
        masked=model_params["masked"],
        num_heads=model_params["heads"],
        max_tokens=model_params["max_tokens"],
    )
    model.load_state_dict(torch.load(f"{name}.pt", weights_only=True))
    return model
        
if __name__ == "__main__":
    
    VOCAB_SIZE = 32
    D_MODEL    = 8
    LAYERS     = 24
    MASKED     = True
    NUM_HEADS  = 1
    MAX_TOKENS = 256
    
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        layers=LAYERS,
        masked=MASKED,
        num_heads=NUM_HEADS,
        max_tokens=MAX_TOKENS,
    )
    print(f"Model Params - {model.returnParams()}")
    save_model("./models/MiniGPT", model)
    del model

    model = load_model("./models/MiniGPT").to("cuda")
    print(model(torch.randint(low=0, high=VOCAB_SIZE, size=(1, VOCAB_SIZE)).to("cuda")))