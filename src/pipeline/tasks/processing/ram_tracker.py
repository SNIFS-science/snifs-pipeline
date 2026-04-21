import os
import threading
import time

import psutil


# This function calculates total RAM used by the script + all workers
def get_total_mem():
    # Identify the main Python process
    main_process = psutil.Process(os.getpid())
    # Start the counter with the main script's memory
    total = main_process.memory_info().rss
    # Find all "child" processes (the CPU workers) and add their memory
    for child in main_process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Ignore processes that finished exactly while we were checking
    return total / (1024 * 1024)  # Convert bytes to Megabytes


# This creates a new thread so I can track the memory usage of the program without stopping the program
class MemoryMonitor(threading.Thread):
    def __init__(self, ticks_per_update=5, tick_len=1):
        super().__init__()
        self.peak_mem = 0
        self.keep_running = False
        self.ram_history = []
        self.ticks = 0

        self.ticks_per_update = ticks_per_update
        self.tick_len = tick_len

        self.bin_for_avg = []

    def run(self):
        # While the pool is working, keep checking memory every 0.1 seconds
        self.keep_running = True

        while self.keep_running:
            current = get_total_mem()
            if current > self.peak_mem:
                self.peak_mem = current

            if self.ticks % self.ticks_per_update == 0:
                self.bin_for_avg.append(current)
                self.ram_history.append(sum(self.bin_for_avg) / len(self.bin_for_avg))
                self.bin_for_avg = []
            else:
                self.bin_for_avg.append(current)

            time.sleep(0.1)  # made a mistake and only took measurements everysecond

            self.ticks += self.tick_len

    def get_peak_mem(self):
        return self.peak_mem

    def get_history(self):
        return self.ram_history

    def stop(self):
        self.keep_running = False

    def get_tick(self):
        return self.tick

    def get_history_index(self):
        return int(self.ticks / self.ticks_per_update)
