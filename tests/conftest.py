import os

import pytest

os.environ.setdefault("TORCHPOTENTIAL_DEVICE", "cpu")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="session")
def data_dir():
    return DATA
