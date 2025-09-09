# Tiny-GPTs
Tiny-GPTs is a simple way to build custom encoder/decoder only models from scratch. Allowing people to better understand how they work. The repository includes ways to pretrain and finetune your LLM, compile datasets into npy files, and create a tokenizer for them to use. unfortunately ways to train via reinforcement learning haven't been included yet but should be added later down the line using [Huggingfaces TRL Library](https://huggingface.co/docs/trl/en/index). 

## How to install
To install the model you need to have all the requirements. The requirements for this model are all contained in [requirements](./requirements.txt). I used a Anaconda environment for my install so I'd recommend you do the same or do something similar. To begin install [PyTorch](https://pytorch.org/get-started/locally/) on your machine, it's important you use at least PyTorch 2.4 or higher. To do this follow the guide found on their website.
Once PyTorch has been installed simply follow the command below.
```
pip install -r requirements.txt
```
This will begin to install all the necessary libraries in order to successfully run Tiny-GPTs. I've also included two simple models that had marginal success on text completion and text generation.

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

## Training your own model
To make training these models as easy possible I've broken down training into two different files. These files are named `Training-pretrain.py` and `Training-finetune.py` the correct way to train your models is to call pretrain first, this is designed to teach the models the basic of text like nouns, verbs, and sentiments for example. The model at this stage won't be suitable for tasks but will have a "background knowledge" in how text should be structured. After this we then fine tune the model for desired traits, these traits can be text generation or filling in missing characters that maybe missing from the string.

#### Building Vocabs
Inside the `datasets` directory I have two files named [simple_tokenizer.py](datasets/simple_tokenizer.py) and [vocab_builder.py](datasets/vocab_builder.py) (As a disclaimer GPT-5 build the main tokenizer. I modified code for suffix and concatenation removal) Vocab building is done rather easily by calling [vocab_builder.py]() in the command line.
```cmd
(pytorch) D:\Python\LLM\dataset>python vocab_builder.py -h
usage: vocab_builder.py [-h] --name NAME [--f F] --csv CSV [--skip SKIP]

vocab_builder.py will build your vocab for you and store it in a json file. It
will only add words if they appear more than F number of times. This way random
phrases or misspellings are more likely to be avoided. It will build its dataset
from a csv dataset file. I recommend using the mini-wiki-text.csv file.

options:
  -h, --help   show this help message and exit
  --name NAME  Name of new vocab
  --f F        Add words if appear more than F. Default is 5.
  --csv CSV    Read from csv dataset
  --skip SKIP  number of rows to skip in dataset. Prevents large dataset
               explosion while keeping the vocab diverse. Default is 10.
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

To prepare dataset call `eda.py` in the command line. 
```cmd
(pytorch) D:\Python\LLM\dataset>python eda.py -h
usage: eda.py [-h] --vocab VOCAB --csv CSV --columns COLUMNS [--name NAME] 
[--length LENGTH] [--hide_rate HIDE_RATE] [--roles ROLES] [--operation OPERATION]

EDA.py Will build your datasets as save them in npy file format. Output format will be the tokenised data with features being the initial inputs and labels the output shifted right. Beginning/End of sequence tags are included by default.

options:
  -h, --help            show this help message and exit
  --vocab VOCAB         The name of the vocab file saved in vocabs
  --csv CSV             Dataset csv file (NO INDEX)
  --columns COLUMNS     Should be either 1 or 2.
  --name NAME           Header name for saved files (default 'Example')
  --length LENGTH       truncate examples to desired length (default 512)
  --hide_rate HIDE_RATE
                        The chance of hiding tokens (default 0.2)
  --roles ROLES         Decides to add roles like OPERATOR/BOT.
  --operation OPERATION

                                Decides which dataset to generate:
                                    0 : Next token prediction
                                    2 : Fill the empty spaces
                                    3 : Generates both.
```

`eda.py` will generate `.npy` files containing a compiled dataset. It will create two files a `NAME-OPERATION-labels.npy` and a `NAME-OPERATION-features.npy` file. The flags will do the following.
```
vocab   - Give the name of the vocab to use in the 'vocab' directory.
csv     - Give the name of the csv dataset you'd like to compile.
columns - The number of columns in csv dataset.
name    - The name for the compiled dataset.
length  - Truncate large texts down to a max size.
hide_rate - Probability to hide a token when masking for completion.
roles   - Enable roles, use for finetuning.
operation - Which operation we're compiling for, text generation, masking or both.
```

You can use custom datasets if you'd like, this tool is just to streamline training if you desire a fast simple solution. I would recommend you compile and build your own datasets.

#### Training your model
For training your model I would HEAVILY recommend renting a GPU from the cloud. I used [Vast.ai](vast.ai) for training my larger models on a RTX 5090 or H200 if required. Smaller models under 30M parameters could be trained at home. Overall I have two files for training models, `Training-pretraining.py` and `Training-finetuning.py` both of these files are practically identical and will be merged in future under one file.  You should run `Training-pretraining.py` first then followed by `Training-finetuning.py` you must change the model to train in `Training-finetuning.py` in order to get satisfactory results. This is only a temporary training solution and will be updated soon enough to be more efficient.
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
