import os

from celery import Celery

path = "./celery/data/broker"
if not os.path.exists(path):
    os.makedirs(path)
path = "./celery/results"
if not os.path.exists(path):
    os.makedirs(path)

app = Celery(
    "experiment",
    broker_url='filesystem://localhost',
    broker_transport_options={
        'data_folder_in': './celery/data/broker',
        'data_folder_out': './celery/data/broker/',
    },
    result_backend='file://./celery/results',
    task_serializer = 'json',
    persist_results = True,
    result_serializer = 'json',
    accept_content = ['json'],
    imports=("tasks",),
)
