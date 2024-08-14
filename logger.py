from collections import defaultdict
import time
import torch

class TimeLogger(object):
    def __init__(self):
        super().__init__()
        self.dict_ts = defaultdict(float)
        self.prev_time = None

    def reset(self):
        self.prev_time = time.time()

    def log(self, key):
        torch.cuda.synchronize()
        now = time.time()
        delta = now - self.prev_time
        self.dict_ts[key] += delta
        self.prev_time = now

    def print_log(self):
        to_print = ", ".join([f"{k}:{self.dict_ts[k] :4.2f}" for k in sorted(self.dict_ts.keys())])
        print(to_print)

    def export_dict(self):
        return self.dict_ts

