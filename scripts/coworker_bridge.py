import time
import subprocess
import os

print("Antigravity Coworker Bridge initialized.")
print("Monitoring Hermes's ~/antigravity_dev/ for the next 30 minutes...")

end_time = time.time() + 1800  # 30 minutes
sync_targets = [
    "core/",
    "antigravity/",
    "live_daemon.py",
    "train.py",
    "data/"
]

while time.time() < end_time:
    for target in sync_targets:
        try:
            cmd = [
                "wsl", "-d", "Ubuntu", "--", "rsync", "-av", 
                "--exclude=__pycache__", 
                f"~/antigravity_dev/{target}", 
                f"./{target}" if target.endswith("/") else f"./{target}"
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # If rsync transferred any files (output length > ~100 chars for just the header/footer), print it
            lines = result.stdout.strip().split('\n')
            if len(lines) > 3:  # Means files were actually updated
                print(f"[{time.strftime('%H:%M:%S')}] Hermes updated {target}:")
                for line in lines[1:-3]: # Skip the header and footer boilerplate
                    if line.strip() and not line.endswith('/'):
                        print(f"  -> {line}")
        except Exception as e:
            pass
            
    time.sleep(60)

print("Coworker Bridge shift completed.")
