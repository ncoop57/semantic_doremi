#!/bin/bash

GLOBAL_RANK=$SLURM_PROCID
CPUS=$SLURM_CPUS_ON_NODE
MEM=$SLURM_MEM_PER_NODE # seems to be in MB
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
LOCAL_IP=$(hostname -I | awk '{print $1}')

export MEMORY=$MEM
export CPUS=$CPUS

ulimit -n 75000

source $(PWD)/venv/bin/activate

# Convert memory to bytes
MEM_BYTES=$((MEM * 1024 * 1024))

# Check if memory is below 75MB
if [ $MEM_BYTES -lt 78643200 ]; then
    echo "Error: Memory is below 75MB. Exiting."
    exit 1
fi

echo "MEM IN BYTES: $MEM_BYTES"

if [ $GLOBAL_RANK -eq 0 ]; then
    # print out some info
    echo -e "MASTER ADDR: $MASTER_ADDR\tGLOBAL RANK: $GLOBAL_RANK\tCPUS PER TASK: $CPUS\tMEM PER NODE: $MEM_BYTES"

    # start the head node
    ray start --head --port=6370 --node-ip-address=$LOCAL_IP --num-cpus=$CPUS --block --resources='{"resource": 100}' --include-dashboard=true --object-store-memory=214748364800
else
    sleep 10

    # start worker nodes
    ray start --address=$MASTER_ADDR:6370 --num-cpus=$CPUS --block --resources='{"resource": 100}' --object-store-memory=214748364800
    echo "Hello from worker $GLOBAL_RANK"
fi

sleep 10000000
