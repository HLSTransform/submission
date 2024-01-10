from codecarbon import track_emissions

import subprocess

subprocess.call(f'make clean', shell=True)
subprocess.call(f'make runfast_benchmark', shell=True)


@track_emissions()
def run_model(tokens):
    subprocess.call(
        f'./runq_power_consumption ./modelq.bin -t 1 -n {tokens} -i ""', shell=True)


def get_latency():
    subprocess.call(
        f'./runq_latency_256 ./modelq.bin -t 1 -n 256 -i ""', shell=True)
    subprocess.call(
        f'./runq_latency_1024 ./modelq.bin -t 1 -n 1024 -i ""', shell=True)


run_model(256)
run_model(1024)

get_latency()


subprocess.call(f'accuracy_benchmark_non_quantized.bash', shell=True)

subprocess.call(f'accuracy_benchmark_non_quantized.bash', shell=True)
