# Benchmarking Llama 2 CPU

This repository contains the benchmarking the power consumption metrics and inference speed metrics for Llama 2; the original repo by Karpathy can be found here : https://github.com/karpathy/llama2.c/ 

To run our benchmarks, install codecarbon via `pip install codecarbon` and run `python benchmark.py`. 

For now, only Linux/Unix systems are supported. Emissions data will be saved in `emissions.csv`; these represent total energy compilations over 100 runs. Latency data is saved in `256_cpu.txt` and `1024_cpu.txt`.

The nonquantized model weights for `stories110M.bin` need to be downloaded (can be found here https://huggingface.co/karpathy/tinyllamas/tree/main) and dragged into this folder. The quantized model can be created by running `python export.py stories110Mq.bin --version 2 --checkpoint stories110M.pt`