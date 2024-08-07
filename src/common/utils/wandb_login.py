import os, subprocess
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

key = os.getenv("WANDB_API_KEY")
command = f"wandb login --host https://wandb.staging.equinoxai.com {key}"
process = subprocess.Popen(command, shell=True)
result = process.communicate()
