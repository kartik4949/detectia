# Copyright 2021 Kartik Sharma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

from absl import logging

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TEST_ROOT)
TEMP_PATH = os.path.join(PROJECT_ROOT, "test_temp")

if PROJECT_ROOT not in os.getenv("PYTHONPATH", ""):
    splitter = ":" if os.environ.get("PYTHONPATH", "") else ""
    os.environ[
        "PYTHONPATH"
    ] = f'{PROJECT_ROOT}{splitter}{os.environ.get("PYTHONPATH", "")}'


if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)
