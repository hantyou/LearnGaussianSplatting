import pytest


def pytest_addoption(parser):
    parser.addoption("--student", action="store_true", help="Test exercises/core.py instead of the reference")


@pytest.fixture
def core(request):
    if request.config.getoption("--student"):
        from exercises import core
    else:
        from splatlab import core
    return core
