#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <machine_type>"
    echo "Valid options: timpc, mac, laptop, white-machine, alienware, lab-comp"
    exit 1
fi

MACHINE=$1
ALGOS=("dqn" "ppo" "sac")
ENVS=("cartpole" "minigrid" "mujoco")

# Define the devices, preambles, and search devices based on the machine
case $MACHINE in
    "white-machine")
        DEV1_NAME="white-machine_gpu0"
        DEV1_PRE="OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11"
        DEV1_SEARCH_DEVS="cpu cuda:0"
        
        DEV2_NAME="white-machine_gpu1"
        DEV2_PRE="OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15"
        DEV2_SEARCH_DEVS="cpu cuda:0"
        ;;
    "alienware")
        DEV1_NAME="alienware_gpu_0"
        DEV1_PRE="OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5"
        DEV1_SEARCH_DEVS="cpu cuda:0"
        
        DEV2_NAME="alienware_gpu_1"
        DEV2_PRE="OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7"
        DEV2_SEARCH_DEVS="cpu cuda:0"
        ;;
    "lab-comp")
        DEV1_NAME="lab-comp_cpu"
        DEV1_PRE="OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=\"\" numactl --cpunodebind=0 --membind=0"
        DEV1_SEARCH_DEVS="cpu"
        
        DEV2_NAME="lab-comp_gpu"
        DEV2_PRE="OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1"
        DEV2_SEARCH_DEVS="cpu cuda:0"
        ;;
    "mac")
        DEV1_NAME="$MACHINE"
        DEV1_PRE=""
        DEV1_SEARCH_DEVS="cpu"
        
        DEV2_NAME=""
        DEV2_PRE=""
        DEV2_SEARCH_DEVS=""
        ;;
    "timpc"|"laptop")
        DEV1_NAME="$MACHINE"
        DEV1_PRE=""
        DEV1_SEARCH_DEVS="cpu cuda"
        
        DEV2_NAME=""
        DEV2_PRE=""
        DEV2_SEARCH_DEVS=""
        ;;
    *)
        echo "Unknown machine type: $MACHINE"
        exit 1
        ;;
esac

echo "Starting concurrent benchmarks for $MACHINE..."

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "====================================================="
        echo "Benchmarking $algo on $env..."
        
        # Launch Device 1 in the background
        eval "$DEV1_PRE python benchmark.py --algo $algo --grid_search --device_name \"$DEV1_NAME\" --env_name $env --search_devices $DEV1_SEARCH_DEVS" &
        PID1=$!
        
        # Launch Device 2 in the background (if it exists)
        if [ -n "$DEV2_NAME" ]; then
            eval "$DEV2_PRE python benchmark.py --algo $algo --grid_search --device_name \"$DEV2_NAME\" --env_name $env --search_devices $DEV2_SEARCH_DEVS" &
            PID2=$!
        fi
        
        # Wait for both processes to finish before moving to the next algo/env combination
        wait $PID1
        if [ -n "$DEV2_NAME" ]; then
            wait $PID2
        fi
    done
done

echo "====================================================="
echo "All grid searches for $MACHINE completed!"