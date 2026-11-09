# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Check serial runner selection and failure propagation without NPUs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_LIBRARY = Path(__file__).resolve().parents[2] / "tests/npu_smoke/lib_npu_smoke.sh"


@pytest.fixture
def serial_runner(tmp_path):
    root = tmp_path / "repo"
    scripts = root / "tests/npu_smoke"
    scripts.mkdir(parents=True)
    shutil.copyfile(_LIBRARY, scripts / _LIBRARY.name)
    commands = tmp_path / "bin"
    commands.mkdir()
    for command in ("ray", "pkill", "sleep", "npu-smi"):
        stub = commands / command
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    # Exercise the shell runner, not allocator behavior; never preload a fake ELF.
    if sys.platform == "darwin":
        preload = "libjemalloc-shell-test"
    else:
        libc = next(path for base in (Path("/lib"), Path("/usr/lib")) for path in base.glob("**/libc.so.6"))
        preload = str(tmp_path / "libjemalloc-shell-test.so")
        Path(preload).symlink_to(libc)

    def run(body, selected="", *, tee_failure=False, args=()):
        script = scripts / "run.sh"
        script.write_text(
            "#!/usr/bin/env bash\nexport NPU_SMOKE_GROUP=core\n"
            'source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"\n' + body + "\nprint_summary\n"
        )
        if tee_failure:
            tee = commands / "tee"
            tee.write_text("#!/bin/sh\ncat >/dev/null\nexit 23\n")
            tee.chmod(0o755)
        env = dict(os.environ)
        for key in ("NUM_NPUS", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
            env.pop(key, None)
        env.update(
            PATH=str(commands) + os.pathsep + os.environ["PATH"],
            LD_PRELOAD=preload,
            LOG_DIR=str(tmp_path / "logs"),
            NPU_SMOKE_TEST_IDS=selected,
        )
        return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=15)

    return run


@pytest.mark.parametrize("exit_code", [0, 1, 23, 134])
def test_workload_status_including_post_success_abort(serial_runner, exit_code):
    body = f"run_test 19 fixture bash -c 'echo \"1 passed\"; exit {exit_code}'"
    result = serial_runner(body)
    assert (result.returncode == 0) == (exit_code == 0), result.stdout + result.stderr
    if exit_code:
        assert f"exit={exit_code}" in result.stdout
        assert "[PASS]" not in result.stdout


def test_failure_does_not_skip_following_entry_or_make_suite_pass(serial_runner):
    result = serial_runner("run_test 1 first false\nrun_test 2 second true")
    assert result.returncode != 0
    assert "[FAIL] [1]" in result.stdout
    assert "[PASS] [2]" in result.stdout
    assert "Failed: 1" in result.stdout


def test_logging_failure_does_not_count_as_pass(serial_runner):
    result = serial_runner("run_test 1 fixture true", tee_failure=True)
    assert result.returncode != 0
    assert "tee_exit=23" in result.stdout
    assert "[PASS]" not in result.stdout


def test_selection_preserves_eight_device_budget(serial_runner):
    result = serial_runner(
        "run_test 0 omitted false\n"
        "run_test 19 selected bash -c 'echo devices=$NUM_NPUS:$ASCEND_RT_VISIBLE_DEVICES:$ASCEND_VISIBLE_DEVICES'",
        selected="19",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] [19]" in result.stdout
    assert "Starting  [0]" not in result.stdout
    assert "devices=8:0,1,2,3,4,5,6,7:0,1,2,3,4,5,6,7" in result.stdout


@pytest.mark.parametrize("selected", ["00", "1,", ",1", "1 2", "1,1", "-1", "x", "99"])
def test_invalid_or_unknown_selection_fails(serial_runner, selected):
    result = serial_runner("run_test 1 fixture true", selected=selected)
    assert result.returncode == 2, result.stdout + result.stderr


def test_duplicate_registration_fails(serial_runner):
    result = serial_runner("run_test 1 fixture true\nrun_test 1 repeated true")
    assert result.returncode == 2
    assert "registered more than once" in result.stdout


def test_cli_device_budget_is_not_capped(serial_runner):
    result = serial_runner("run_test 1 fixture bash -c 'echo count=$NUM_NPUS'", args=("--num-npus", "4"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "count=4" in result.stdout
