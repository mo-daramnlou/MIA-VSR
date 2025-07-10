import GPUtil
import psutil
import threading
import time
import subprocess

def monitor_gpu(delay=20):
    """Prints GPU utilization and memory every 'delay' seconds."""
    print("Monitoring GPU... Press Ctrl+C to stop.")
    try:
        while True:
            GPUtil.showUtilization()
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            mem = psutil.virtual_memory()
            print(f"CPU Usage (per core): {cpu_percent}%")
            print(f"Memory Usage: {mem.percent}% ({mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB)")
            print('='*80)
            time.sleep(delay)
    except KeyboardInterrupt:
        print("GPU monitoring stopped.")

# You can call this in a separate thread or at intervals during training
# Example (simple call, will block training):

monitor_gpu()
# monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
# monitor_thread.start()