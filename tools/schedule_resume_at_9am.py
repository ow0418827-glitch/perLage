"""明晨 09:00 精准守护与拉起脚本：
守护至 2026-11-10 09:00:00，准点拉起 tools/run_perlage_no_pixel_groups9_20261109.py
"""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))

PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
RUN_SCRIPT = PROJECT_ROOT / "tools" / "run_perlage_no_pixel_groups9_20261109.py"
LOG_DIR = PROJECT_ROOT / "output" / "perlage_no_pixel_groups9_20261109"
LOG_FILE = LOG_DIR / "scheduled_run.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TIME = datetime.datetime(2026, 11, 10, 9, 0, 0)

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"[{datetime.datetime.now()}] 定时守护进程已启动，计划于 {TARGET_TIME} 执行续作\n")
    f.flush()

while True:
    now = datetime.datetime.now()
    if now >= TARGET_TIME:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{now}] === 达到目标时间 09:00:00！立即在断点处拉起实验流水线 ===\n")
            f.flush()
        break
    remaining = (TARGET_TIME - now).total_seconds()
    if remaining > 60:
        time.sleep(min(remaining - 30, 30))
    else:
        time.sleep(1)

with open(LOG_FILE, "a", encoding="utf-8") as f:
    proc = subprocess.Popen(
        [str(PYTHON_EXE), "-u", str(RUN_SCRIPT)],
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    f.write(f"[{datetime.datetime.now()}] 实验子进程已启动 (PID={proc.pid})\n")
    f.flush()
    proc.wait()
    f.write(f"[{datetime.datetime.now()}] 实验子进程执行完毕，退出码={proc.returncode}\n")
    f.flush()
