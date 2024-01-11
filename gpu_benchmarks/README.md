# Llama 2

This repository contains the code for benchmarking gpu energy and latency metrics for Llama 2; this repo is a fork of the orignal implementation repo in https://github.com/facebookresearch/llama.


To run our benchmarks, install the dependencies found in the original Llama 2 repo. Then, get the weights which can be found here (https://huggingface.co/karpathy/tinyllamas/tree/main) [we use stories110M.pt], install codecarbon via `pip install codecarbon`, and run `torchrun tinyllama2_power_consumption.py` and `torchrun tinyllama2.py` for energy/power consumption metrics and latency metrics respectively.