"""Project-owned artifacts and portable, shared input locations."""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configured_path(variable, fallback):
    load_dotenv(PROJECT_ROOT / '.env')
    return Path(os.getenv(variable, str(fallback))).expanduser().resolve()


def dataset_storage_root():
    return configured_path('TIME_DATASET', configured_path('TIME_DATA_ROOT', PROJECT_ROOT / 'datasets') / 'hf_dataset')


def weights_root():
    return configured_path('TIME_WEIGHTS', PROJECT_ROOT / 'weights')


def artifact_project_root():
    load_dotenv(PROJECT_ROOT / '.env')
    if os.getenv('SELENA_NNI'):
        return Path(os.environ['TIME_STORAGE_ROOT']).expanduser().resolve() / 'codes' / 'selectime'
    return PROJECT_ROOT


def outputs_root():
    return artifact_project_root() / 'outputs'


def logs_root():
    return artifact_project_root() / 'logs'
