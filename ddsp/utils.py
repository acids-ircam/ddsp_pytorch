import numpy as np


def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule