from GPT import GPT, save_model

from argparse import RawTextHelpFormatter
import argparse

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def args_builder():
    parser = argparse.ArgumentParser(
    description="""
    Model-Builder.py is responsible for building your model
    in a way that allows for easy training in the training scripts.
    It will by default save your models to the 'models' subfolder.
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--vocab",
        type=int,
        required=True,
        help="The vocab size of the model."
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True,
        help="The models dimension."
    )
    parser.add_argument(
        "--layers",
        type=int,
        required=True,
        help="The number of transformer layers."
    )
    parser.add_argument(
        "--masked",
        type=bool,
        required=False,
        default=True,
        help="Masked attention or not (Encoder vs Decoder, default is Decoder True)."
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="The number of attention heads (MUST BE A FACTOR OF d_model)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=False,
        default=512,
        help="The maximum context window of the model (Defualt 512)."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="My-Model",
        help="The name of the model."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_builder()
    VOCAB_SIZE = args.vocab
    D_MODEL    = args.d_model
    LAYERS     = args.layers
    MASKED     = args.masked
    NUM_HEADS  = args.num_heads
    MAX_TOKENS = args.max_tokens
    MODEL_NAME = args.name
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        layers=LAYERS,
        masked=MASKED,
        num_heads=NUM_HEADS,
        max_tokens=MAX_TOKENS
    )
    save_model(f"models/{MODEL_NAME}", model)
    print("Model saved without issue.")