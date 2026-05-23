import pytest


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests that need real GPU + ComfyUI + vLLM"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="need --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
