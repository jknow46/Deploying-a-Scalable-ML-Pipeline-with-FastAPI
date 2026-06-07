import pytest
import pandas as pd
import os

os.environ["WANDB_MODE"] = "offline"
os.chdir(os.getcwd())

def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")

def _resolve_path(filename: str) -> str:
    return os.path.join(os.getcwd(), filename)

@pytest.fixture(scope='session')
def data(request):
    csv_name = request.config.option.csv
    if csv_name is None:
        pytest.fail("You must provide the --csv option on the command line")

    csv_path = _resolve_path(csv_name)
    if not os.path.exists(csv_path):
        pytest.fail(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path)

@pytest.fixture(scope='session')
def ref_data(request):
    ref_name = request.config.option.ref
    if ref_name is None:
        pytest.fail("You must provide the --ref option on the command line")

    ref_path = _resolve_path(ref_name)
    if not os.path.exists(ref_path):
        pytest.fail(f"Reference CSV file not found: {ref_path}")

    return pd.read_csv(ref_path)

@pytest.fixture(scope='session')
def kl_threshold(request):
    value = request.config.option.kl_threshold
    if value is None:
        pytest.fail("You must provide a threshold for the KL test")
    return float(value)

@pytest.fixture(scope='session')
def min_price(request):
    value = request.config.option.min_price
    if value is None:
        pytest.fail("You must provide min_price")
    return float(value)

@pytest.fixture(scope='session')
def max_price(request):
    value = request.config.option.max_price
    if value is None:
        pytest.fail("You must provide max_price")
    return float(value)