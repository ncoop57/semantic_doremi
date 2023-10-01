import submitit
import os
import time
import argparse

def run_ray_worker(global_rank, master_addr, cpus, mem_bytes, venv_path):
    local_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()

    os.system(f"ulimit -n 75000")
    os.system(f"source {venv_path}/bin/activate")

    if global_rank == 0:
        print(f"MASTER ADDR: {master_addr}\tGLOBAL RANK: {global_rank}\tCPUS PER TASK: {cpus}\tMEM PER NODE: {mem_bytes}")
        os.system(f"ray start --head --port=6370 --node-ip-address={local_ip} --num-cpus={cpus} --block --resources='{{\"resource\": 100}}' --include-dashboard=true --object-store-memory=214748364800")
    else:
        time.sleep(10)
        os.system(f"ray start --address={master_addr}:6370 --num-cpus={cpus} --block --resources='{{\"resource\": 100}}' --object-store-memory=214748364800")
        print(f"Hello from worker {global_rank}")

    time.sleep(10000000)

def start_ray_cluster(venv_path):
    global_rank = os.environ["SLURM_PROCID"]
    cpus = os.environ["SLURM_CPUS_ON_NODE"]
    mem = int(os.environ["SLURM_MEM_PER_NODE"])  # in MB
    master_addr = os.popen('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1').read().strip()
    mem_bytes = mem * 1024 * 1024

    if mem_bytes < 78643200:
        print("Error: Memory is below 75MB. Exiting.")
        exit(1)

    run_ray_worker(global_rank, master_addr, cpus, mem_bytes, venv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Ray cluster on Slurm using submitit.")
    parser.add_argument("--venv_path", type=str, required=True, help="Path to the virtual environment directory")
    parser.add_argument("--nodes", type=int, default=12, help="Number of nodes for the Slurm job")
    parser.add_argument("--partition", type=str, default="g40", help="Slurm partition name")
    parser.add_argument("--mem_gb", type=int, default=999, help="Memory in GB per node for the Slurm job")
    parser.add_argument("--job_name", type=str, default="applied_ablation", help="Name of the Slurm job")
    parser.add_argument("--account", type=str, default="stablegpt", help="Slurm account name")

    args = parser.parse_args()

    # Configure the Slurm job using submitit
    executor = submitit.AutoExecutor(folder="raylogs")
    executor.update_parameters(
        partition=args.partition,
        job_name=args.job_name,
        nodes=args.nodes,
        tasks_per_node=1,
        exclusive=True,
        mem_gb=args.mem_gb,
        account=args.account
    )

    job = executor.submit(start_ray_cluster, args.venv_path)
    print(f"Ray Cluster started with Job ID: {job.job_id}")
