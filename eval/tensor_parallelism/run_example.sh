
# To run samples:
# bash run_example.sh {file_to_run.py} {num_gpus}
# num_gpus = num local gpus to use (must be at least 2). Default = 2

# samples to run include:
# tensor_parallel_example.py

echo "Launching ${1:-tensor_parallel_example.py} with ${2:-2} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-4} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-tensor_parallel_example.py}
