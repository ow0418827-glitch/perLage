import subprocess
from pathlib import Path

bat = Path("tools/run_perlage_resume_scheduled.bat").resolve()
cmd = [
    "schtasks", "/Create",
    "/SC", "ONCE",
    "/TN", "PerLAGE_Resume_9AM",
    "/TR", f'"{bat}"',
    "/ST", "09:00",
    "/SD", "2026/11/10",
    "/F"
]
res = subprocess.run(cmd, capture_output=True, text=True)
print("RET:", res.returncode)
print("OUT:", res.stdout)
print("ERR:", res.stderr)
