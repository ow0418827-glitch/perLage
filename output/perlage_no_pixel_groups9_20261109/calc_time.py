from datetime import datetime, timedelta
now = datetime.now()
target = datetime(now.year, now.month, now.day, 9, 0, 0)
if target <= now:
    target += timedelta(days=1)
diff = (target - now).total_seconds()
print("NOW:", now.strftime("%Y-%m-%d %H:%M:%S"))
print("TARGET:", target.strftime("%Y-%m-%d %H:%M:%S"))
print("DIFF_SECONDS:", int(diff))
print("DIFF_HOURS:", round(diff / 3600, 2))
