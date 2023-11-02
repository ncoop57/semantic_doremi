import fsspec
from tqdm import tqdm
import ray
# import ray
import os
import pandas as pd
import numpy as np
from transformers import AutoModel
import torch
import time

ray.init(address="auto", ignore_reinit_error=True) # , _temp_dir='/admin/home-nathan/semantic_doremi/ray_tmp/')
# ray.init(include_dashboard=True, dashboard_host='0.0.0.0', dashboard_port=8265)

fs = fsspec.filesystem("s3")
file_list = fs.glob("s3://pile-everything-west/redpajama_raw/c4/*.jsonl")
file_list = ['s3://' + string for string in file_list]
model_name = "jinaai/jina-embeddings-v2-small-en"

@ray.remote(num_gpus=1,num_cpus=6,max_retries=10)
def worker(file_list):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to('cuda')
    model = torch.compile(model)
    out_fs = fsspec.filesystem(
        "s3",
        config_kwargs={
            "retries": {"max_attempts": 10},
            "max_pool_connections": 512
        }
    )   

    for x in tqdm(file_list, desc="Files", position=0):
        if out_fs.exists(x.replace("redpajama_raw","redpajama_processed")):
            print("File already exists: " + x)
            continue
        with out_fs.open(x, "rb") as f:
            df = pd.read_json(f, lines=True)
        batch_size = 768 # 16384, 32768, 65536
        n = len(df)
        splits = n // batch_size
        batches = np.array_split(df, splits)
        all_embeddings = []
        for batch in tqdm(batches, desc="Batches", position=1, leave=False):
            try:
                torch.cuda.empty_cache()
                embeddings = model.encode(batch["text"].tolist(), max_length=1024, batch_size=batch_size)
            except Exception as e:
                print(e)
                slurm_job_id = os.environ["SLURM_JOB_ID"]
                # dump the batch to a txt file
                with open(f"error_{slurm_job_id}.txt", "w") as f:
                    f.write(str(batch['text'].tolist()))
                raise ValueError(f"Error with batch at {x}")
            all_embeddings.extend(embeddings)
        df['embeddings'] = all_embeddings
        with out_fs.open(x.replace("redpajama_raw","redpajama_processed"), "wb") as f:
            df.to_parquet(f)
        print("Done with file: " + x)

file_list.sort()
# filter out files that have already been processed
processed_fs = fsspec.filesystem(
    "s3",
    config_kwargs={
        "retries": {"max_attempts": 10},
        "max_pool_connections": 512
    }
)
processed_file_list = processed_fs.glob("s3://pile-everything-west/redpajama_processed/c4/*.jsonl")
processed_file_list = ['s3://' + string.replace("redpajama_processed","redpajama_raw") for string in processed_file_list]
processed_file_list.sort()
print(f"Processing {len(file_list)} files")
file_list = np.setdiff1d(file_list, processed_file_list)
print(f"Processed {len(processed_file_list)} files")
print(f"Processing {len(file_list)} files")
gpus = 8
nodes = 4
partitions = gpus * nodes # 8 is gpus and 1 is nodes

file_list = np.array_split(file_list, partitions)

workers = [worker.remote(x) for x in file_list]

ray.get(workers)