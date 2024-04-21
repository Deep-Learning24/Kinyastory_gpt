'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import argparse
import os
import random
import sys

import numpy as np
import torch

from Decoder import Transformer

from utils.sample import sample_sequence
from transformers import AutoTokenizer
from tokenizer_utils import Tokenizer
from utils.utils import load_weight



def text_generator(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=512)
    model_config = {
        'vocab_size': len(tokenizer.get_vocab()),
        'd_model': 12288 // 8,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1
    }
    model = Transformer(**model_config).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    max_length = 512
 
    # Load Model
    # model_path = 'models'
    # trainer = Trainer(model, optimizer, criterion, model_path, model_config)
   
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = max_length
    elif args.length > max_length:
        raise ValueError("Can't get samples longer than window size: %s" % max_length)

    print(args.text)

    tokenizer_instance = Tokenizer(tokenizer)
    handel_encode = tokenizer_instance.handel_encode
    handel_decode = tokenizer_instance.handel_decode

    context_tokens = handel_encode(args.text)[0]

    generated = 0
    start_token = tokenizer.encode("<|endoftext|>")[0]
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens if not args.unconditional else None,
            start_token=start_token if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        #print(out)
        for i in range(args.batch_size):
            generated += 1
            text = handel_decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)


if __name__ == '__main__':
    model_path = 'models'
    model_file = os.path.join(model_path, 'best_gpt2_model.pt')
    if os.path.exists(model_file):
        state_dict = torch.load(model_file,
                                map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
    else:
        print('Please train the model first. Run train.py to train the model.')
        sys.exit()
