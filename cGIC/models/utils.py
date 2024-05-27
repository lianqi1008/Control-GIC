import math
import torch
from functools import partial

# step scheduler
def fn_LinearWarmup(warmup_steps, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def Scheduler_LinearWarmup(warmup_steps):
    return partial(fn_LinearWarmup, warmup_steps)