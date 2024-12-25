import os

from celery import Celery

path = "./celery/data/broker"
if not os.path.exists(path):
    os.makedirs(path)
path = "./celery/results"
if not os.path.exists(path):
    os.makedirs(path)

artifact_dir = "./celery/artifacts"
os.makedirs(artifact_dir, exist_ok=True)

app = Celery(
    "experiment",
    broker="redis://127.0.0.1:6379/0",
    backend='db+postgresql://username:password@127.0.0.1:5432/postgres',
    imports=("src.tasks.build","src.tasks.destroy","src.tasks.run",),
    task_track_started=True,
)
