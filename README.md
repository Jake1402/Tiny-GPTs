# Torch-GPTs
Torch-GPTs is a simple way to build custom encoder/decoder only models from scratch. Allowing people to better understand how they work. The repository includes ways to pretrain and finetune your LLM, compile datasets into npy files, and create a tokenizer for them to use. unfortunately ways to train via reinforcement learning haven't been included yet but should be added later down the line using [Huggingfaces TRL Library](https://huggingface.co/docs/trl/en/index). 

## How to install
To install the model you need to have all the requirements. The requirements for this model are all contained in [requirements](./requirements.txt). I used a Anaconda environment for my install so I'd recommend you do the same or do something similar. To begin install [PyTorch](https://pytorch.org/get-started/locally/) on your machine, it's important you use at least PyTorch 2.4 or higher. To do this follow the guide found on their website.
Once PyTorch has been installed simply follow the command below.
```
pip install -r requirements.txt
```
This will begin to install all the necessary libraries in order to successfully run Torch-GPTs. I've also included two simple models that had marginal success on text completion and text generation.

## Creating/Saving/Loading the model
The file GPT contains three items. These are; the class `GPT`, a function called `save_model(name, model)`, and a function called `load_model(name)`. The class `GPT` is the `nn.Module` that allows users to build a GPT style transformer (encoder or decoder only). It requires the following parameters:
```python
GPT(
        vocab_size : int,  # The vocab size for the model e.g. 512
        d_model : int,     # The dimension size of the model
        layers : int,      # The number of blocks in the model
        masked : int,      # Masking the attention (True is decoder)
        num_heads : int,   # Number of attention heads (d_model % num_heads = 0)
        max_tokens : int,  # Contect length of the model.
) 
```

The next two functions are responsible for saving and loading the GPT models. Saving the model is simple and only requires the user to call `save_model(name, model)` in order to successfully save the model for future use.
```python
save_model(
	name : str,  # The file path and name e.g. models/mini-GPT
	model : GPT  # The model you would like to save
)
```

Loading the model is also very easy and once again only requires a single function call to load the model. 
```python
load_model(
	name : str  # The file path and name e.g. models/mini-GPT
) -> GPT        # Type hinting, the function returns a GPT model.
```
As can be seen from above creating, saving and loading has been made as simple as possible. 

The GPT class can also be imported into different projects and trained for those purposes too including for [ViT](https://arxiv.org/abs/2010.11929) if required. The models `forward` method also include extra parameters to allow for better interactions these include.
```python
GPT.forward(
	inputs : Torch.tensor, # The inputs to the model
	train : bool,          # Model is in training or not
	temperature : float,   # The temperature used for sampling from outputs
	top_k : int            # The topk outputs only.
)
```
## Training your own model
To make training models as easy possible I've broken down training into several different phases. These phases are as follows:
- Building the model with `Builder-Models.py`
- Building the model vocab with `Builder-Vocab.py`
- Building the dataset with `Builder-ds.py`
- Training the model with either `Training-Pytorch-SingleGPU.py` or `Training-Lightning-MultiGPU.py`.

### Building Model
The building model is crucial in allowing for flexibility, this phase lets the user choose the number of layers, vocab size, embedding dimension, and even the type of transformer (Encoder or Decoder only). To run [Builder-Models.py](Builder-Models.py) in the command line type `python Builder-Models.py -h`
```cmd
usage: Builder-Models.py [-h] --vocab VOCAB --d_model D_MODEL --layers LAYERS 
[--masked MASKED] --num_heads NUM_HEADS [--max_tokens MAX_TOKENS] [--name NAME]

    Model-Builder.py is responsible for building your model
    in a way that allows for easy training in the training scripts.
    It will by default save your models to the 'models' subfolder.


options:
  -h, --help            show this help message and exit
  --vocab VOCAB         The vocab size of the model.
  --d_model D_MODEL     The models dimension.
  --layers LAYERS       The number of transformer layers.
  --masked MASKED       Masked attention or not (Encoder vs Decoder, default is Decoder True).
  --num_heads NUM_HEADS
                        The number of attention heads (MUST BE A FACTOR OF d_model)
  --max_tokens MAX_TOKENS
                        The maximum context window of the model (Defualt 512).
  --name NAME           The name of the model.
```

Upon entering the desired parameters, the script will generate a `.pt` and `.json` file containing the model weights and the python parameters to load the model with the built in `load_model(str : name) -> GPT` function as discussed above.
#### Building Vocabs
[Tokeniser.py](Tokeniser.py) needs a suitable vocab, we generate these using [Builder-Vocab.py](Builder-Vocab.py) (As a disclaimer GPT-5 built the initial tokenizer which was then modified for better efficiency) Vocab building is done rather easily by calling [Builder-Vocab.py](Builder-Vocab.py) in the command line.
```cmd
(pytorch) D:\Python\Torch-GPTs>python Builder-Vocab.py -h
usage: Builder-Vocab.py [-h] --name NAME [--f F] --csv CSV [--skip SKIP]

Builder-Vocab.py will build your vocab for you and store it in a json file. It will only add words if they appear more than F number of times. This way random phrases or misspellings are more likely to be avoided. It will build its
dataset from a csv dataset file. I recommend using the mini-wiki-text.csv file.

options:
  -h, --help   show this help message and exit
  --name NAME  Name of new vocab
  --f F        Add words if appear more than F. Default is 5.
  --csv CSV    Read from csv dataset
  --skip SKIP  number of rows to skip in dataset. Prevents large dataset explosion                while keeping the vocab diverse. Default is 10.
```

This will generate a `NAME-vocab.json` file in the vocabs folder on top of generating a vocab it will also scrub out any non unicode characters and remove any html tags. The first 8 indexes in your vocab will be special tokens and words, these special tokens are.
```
<bos>     - Beginning of sequence.
<eos>     - End of sequence.
<unk>     - Unknown token.
<pad>     - Padding token, used for batching.
operator  - Signals the user is speaking.
user      - Signals the user is speaking.
bot       - Signals the model is speaking.
agent     - Signals the model is speaking.
```

#### Preparing Data
Preparing data has been made very easy simple and some datasets have already been included for people to use. Your dataset should be in the following format for pretraining:
```CSV
TEXT,
Rome was built in a day.,
This movie was very boring.,
```
And finetuning should Ideally be in this format.
```csv
INPUT, RESPONSE,
Hello!, Hello, how are you?,
What's the capital of Italy?, Rome.,
```
Datasets should be in `csv` format and no feature an index column one or two columns. An example of a pretraining dataset would be the `mini-wiki-texts.csv` or the `movie-reviews.csv` datasets. As both of these datasets don't include a second column yet can still teach a model to learn text and how sentences should be structured. An example of a finetuning dataset would be the `instruction-texts.csv` as the dataset features an input and response column.

To prepare dataset call Builder-DS.py` in the command line. 
```cmd
usage: Builder-DS.py [-h] --vocab VOCAB --csv CSV --columns COLUMNS [--name NAME] [--length LENGTH] [--hide_rate HIDE_RATE] [--roles ROLES] [--pad PAD] [--operation OPERATION] [--tags TAGS]

ds_builder.py Will build your datasets as save them in npy file format.
Output format will be the tokenised data with features being the
initial inputs and labels the output shifted right. Beginning/End
of sequence tags are included by default.

options:
  -h, --help            show this help message and exit
  --vocab VOCAB         The name of the vocab file saved in vocabs
  --csv CSV             Dataset csv file (NO INDEX)
  --columns COLUMNS     Should be either 1 or 2.
  --name NAME           Header name for saved files (default 'Example')
  --length LENGTH       truncate examples to desired length (default 512)
  --hide_rate HIDE_RATE
                        The chance of hiding tokens (default 0.2)
  --roles ROLES         Decides to add roles like OPERATOR/AGENT
  --pad PAD             Decides pad sequences to max length default true
  --operation OPERATION
                        Decides which dataset to generate:
                                0 : Next token prediction
                                1 : Fill the empty spaces
                                2 : Generates both.

  --tags TAGS           Decides if <eos>/<bos> tags are added:
                                0 : None
                                1 : <bos> (added to front of user prompt)
                                2 : <eos> (added to end of AI text)
                                3 : <bos> and <eos> in 1/2 positions.
                                Default is None are added
```

`Builder-DS.py` will generate `.npy` files containing a compiled dataset. It will create two files a `NAME-OPERATION-labels.npy` and a `NAME-OPERATION-features.npy` file. You can use custom datasets if you'd like, this tool is just to streamline training if you desire a fast simple solution. I would recommend you compile and build your own datasets.

#### Training your model
For training your model I would HEAVILY recommend renting a GPU from the cloud particularly if you're using larger datasets and models. Smaller models under 30M parameters could be trained at home. Overall I have two files for training models, [Training-Pytorch-SingleGPU.py](Training-Pytorch-SingleGPU.py) and [Training-Lightning-MultiGPU.py](Training-Lightning-MultiGPU.py). Both of these files take the same parameters as input except for the Lightning script which has an extra parameter named `gpus` which takes in the value for the number of GPUs on your system. An important note to remember is  [Training-Lightning-MultiGPU.py](Training-Lightning-MultiGPU.py) was designed with cloud training on Linux systems, as a result if you have any problems in using the script I'd recommend changing `DDPStrategy` backend to `nccl` or any other strategy.

```cmd
usage: Training-PREFFERED-METHOD.py [-h] --model MODEL --vocab VOCAB --training_set TRAINING_SET --gpus GPUS [--epochs EPOCHS] [--batch BATCH] [--lr LR] [--grad_norm GRAD_NORM]

    Training-Lightning-MultiGPU.py is responsible for training your models by
    utilising multiple GPUs. This is done by utilising PyTorch Lightning for safely
    training on multiple GPUs.

options:
  -h, --help            show this help message and exit
  --model MODEL         The name of your generated model.
  --vocab VOCAB         The name of your vocab file.
  --training_set TRAINING_SET
                        The name of your training set (files must end in                                   Feature/Label.npy DON'T INCLUDE)
  --gpus GPUS           Number of GPUs to use during training (Default 1).
  --epochs EPOCHS       Number of epochs for training (Default 30).
  --batch BATCH         The batch size of your training data (Default 64).
  --lr LR               Learning rate for the model (Default 1e-3).
  --grad_norm GRAD_NORM
                        Gradient clipping to stabilise gradients (Default 1.0).
```

During training models will be saved at an interval of every 5 epochs.

### Running Models

Models can be run via the [Run-Blank-Completion.py](Run-Blank-Completion.py) script, this will run the models in a small environment. The models, however, will have no context as context gets wiped after every message sent to the model.

```cmd
usage: Run-Blank-Completion.py [-h] --model MODEL --vocab VOCAB

    Run-Blank-Completion.py is an interface to allow the user to interact with
    there model. It requires two arguments `model` and `vocab`.


options:
  -h, --help     show this help message and exit
  --model MODEL  The name of your generated model.
  --vocab VOCAB  The name of your models vocab.
```

# Final Remarks and References
To finish off I started this project to better understand how LLMs and transformers work. To that I say this project was an overwhelming success, I've learned so much from tinkering with these LLMs in the last few weeks. 

#### Future Plans
My future plans and improvements include:
- Adding RL to the project using Huggingface TRL.
- Adding a DDP option to spread training to other GPUs.
- Cleaning up the code and removing 'spaghetti'.
- Allow the code to be ran purely through command lines rather than editing variables.

#### References
[Attention is all you need](https://arxiv.org/abs/1706.03762) The paper that propose the transformer.<br />
[GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) The paper that proposed GPT style models.<br />
[LLMs are few shot learners](https://arxiv.org/abs/2005.14165) The paper showing scalability of GPTs<br />
[StatQuests transformer](https://www.youtube.com/watch?v=C9QSpl5nmrY) This video helped a lot, especially early on. Fantastic must watch.<br />
