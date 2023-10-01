import submitit
import argparse
from pathlib import Path

def run_embedding(args):
    cmd = f'python embedding.py --max_features {args.max_features} --dataset {args.dataset}'
    return cmd

def run_clustering(args):
    cmd = f'python clustering.py --n_components {args.n_components} --n_clusters {args.n_clusters}'
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launcher for Slurm jobs.")
    parser.add_argument("--job", type=str, choices=["embedding", "clustering"], required=True, help="Which job to run: embedding or clustering.")
    parser.add_argument('--max_features', type=int, default=5000, help="Maximum number of features for TF-IDF.")
    parser.add_argument('--dataset', type=str, help="Name of the huggingface dataset to load.")
    parser.add_argument('--n_components', type=int, default=50, help="Number of components for UMAP reduction.")
    parser.add_argument('--n_clusters', type=int, default=100, help="Number of clusters for KMeans in FAISS.")
    
    # Slurm-specific parameters
    parser.add_argument('--time', type=int, default=120, help="Max runtime in minutes for Slurm job.")
    parser.add_argument('--cpus_per_task', type=int, default=2, help="Number of CPUs per task for Slurm job.")
    parser.add_argument('--tasks_per_node', type=int, default=1, help="Number of tasks per node for Slurm job.")
    parser.add_argument('--gpus_per_node', type=int, default=2, help="Number of GPUs per node for Slurm job.")
    parser.add_argument('--nodes', type=int, default=1, help="Number of nodes for Slurm job.")
    parser.add_argument('--mem_gb', type=int, default=32, help="Memory in GB for Slurm job.")
    parser.add_argument('--slurm_partition', type=str, default="main", help="Slurm partition name.")
    
    args = parser.parse_args()

    # Configure the Slurm job using the parsed arguments
    executor = submitit.AutoExecutor(folder=Path("./logs/{job}/{job}_logs"))
    executor.update_parameters(
        time=args.time,
        cpus_per_task=args.cpus_per_task,
        tasks_per_node=args.tasks_per_node,
        gpus_per_node=args.gpus_per_node,
        nodes=args.nodes,
        mem_gb=args.mem_gb,
        slurm_partition=args.slurm_partition,
        slurm_additional_parameters={"gres": f"gpu:{args.gpus_per_node}"},
        slurm_script=Path("./job_template.sh"),
    )

    if args.job == "embedding":
        cmd = run_embedding(args)
    elif args.job == "clustering":
        cmd = run_clustering(args)

    job = executor.submit(lambda: os.system(cmd))
    print(f"Job {args.job} submitted with ID {job.job_id}")
