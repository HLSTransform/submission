from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import os
from llama import Llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

import numpy as np

import torch
import time

from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, TypedDict


# SET THESE
gen_length = 1024
num_runs = 100
# --------


start_time = time.time()

torch.manual_seed(42)

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
initialize_model_parallel(1)

tokenizer_path = "tokenizer.model"
tokenizer = Tokenizer(model_path=tokenizer_path)

# disable end token
tokenizer.eos_id = None

model_path = "./stories110M.pt"
model_pt = torch.load(model_path)

del model_pt['model_args']['dropout']
model_args = ModelArgs(max_batch_size=32,
                       **model_pt['model_args'])

torch.set_default_tensor_type(torch.cuda.HalfTensor)

assert tokenizer.n_words == model_pt['model_args']['vocab_size']


model = Transformer(model_args)
model.load_state_dict(model_pt['model'], strict=False)
model = Llama(model, tokenizer)


print(f"Loaded in {time.time() - start_time:.2f} seconds")


runs = []

for _ in tqdm(range(num_runs)):
    start_inference_time = time.time()

    gens = model.text_completion(
        prompts=[""], temperature=1, top_p=1, max_gen_len=gen_length)

    # batch size should just be 1
    assert len(gens) == 1

    end_inference_time = time.time()

    num_tokens = len(tokenizer.encode(gens[0]['generation'], False, True))

    time_taken = end_inference_time - start_inference_time

    runs.append(num_tokens/time_taken)


print(np.mean(runs), np.std(runs))

with open(f"log_{gen_length}.txt", 'w') as f:
    for i in runs:
        print(i, file=f)
