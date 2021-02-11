import os

from absl import logging


TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TEST_ROOT)
TEMP_PATH = os.path.join(PROJECT_ROOT, 'test_temp')

if PROJECT_ROOT not in os.getenv('PYTHONPATH', ""):
    splitter = ":" if os.environ.get("PYTHONPATH", "") else ""
    os.environ['PYTHONPATH'] = f'{PROJECT_ROOT}{splitter}{os.environ.get("PYTHONPATH", "")}'


if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)
