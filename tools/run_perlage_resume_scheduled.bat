@echo off
cd /d "d:\Material\Literature Reading\code\perLage"
echo [%date% %time%] 09:00 定时续作启动 >> output\perlage_no_pixel_groups9_20261109\scheduled_run.log
"d:\Material\Literature Reading\code\perLage\.venv\Scripts\python.exe" -u tools\run_perlage_no_pixel_groups9_20261109.py >> output\perlage_no_pixel_groups9_20261109\scheduled_run.log 2>&1
