from codecarbon import track_emissions

import subprocess

subprocess.call(f'make clean', shell=True)
subprocess.call(f'make runfast_benchmark', shell=True)


@track_emissions()
def run_model(tokens):
    subprocess.call(
        f'./runq_power_consumption ./modelq.bin -t 1 -n {tokens} -i ""', shell=True)


def get_latency():
    for i in range(100):
        subprocess.call(
            f'./runq_latency_256 ./modelq.bin -t 1 -n 256 -i ""', shell=True)
    for i in range(100):
        subprocess.call(
            f'./runq_latency_1024 ./modelq.bin -t 1 -n 1024 -i ""', shell=True)


run_model(256)
run_model(1024)

get_latency()


subprocess.call(f'bash accuracy_benchmark_non_quantized.sh', shell=True)

subprocess.call(f'bash accuracy_benchmark.bash', shell=True)
