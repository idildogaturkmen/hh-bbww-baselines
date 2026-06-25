'''
This script lists the number of parquet files for each sample in the collide-1m dataset.
'''
from collections import defaultdict
from huggingface_hub import HfApi

REPO_ID = "fastmachinelearning/collide-1m"
api = HfApi()

files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
parquets = [f for f in files if f.endswith(".parquet")]

by_sample = defaultdict(list)
for f in parquets:
    sample = f.split("/")[0]
    by_sample[sample].append(f)

print("sample,n_parquet_files")
for sample in sorted(by_sample):
    print(f"{sample},{len(by_sample[sample])}")