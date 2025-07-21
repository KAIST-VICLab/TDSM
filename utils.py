import os
import sys
import numpy as np
from pathlib import Path


def write(log, str):
    sys.stdout.flush()
    log.write(str + '\n')
    log.flush()


class Report():
    def __init__(self, save_dir, type):
        filename = os.path.join(save_dir, f'{type}_log.txt')

        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(filename):
            self.logFile = open(filename, 'a')
        else:
            self.logFile = open(filename, 'w')

    def write(self, str):
        print(str)
        write(self.logFile, str)

    def __del__(self):
        self.logFile.close()


class Train_Report():
    def __init__(self):
        self.total_loss = []
        self.diff_loss = []
        self.triplet_loss = []
        self.num_examples = 0

    def update(self, batch_size, total_loss, diff_loss, triplet_loss):
        self.num_examples += batch_size
        self.total_loss.append(total_loss * batch_size)
        self.diff_loss.append(diff_loss * batch_size)
        self.triplet_loss.append(triplet_loss * batch_size)

    def compute_mean(self):
        self.total_loss = np.sum(self.total_loss) / self.num_examples
        self.diff_loss = np.sum(self.diff_loss) / self.num_examples
        self.triplet_loss = np.sum(self.triplet_loss) / self.num_examples

    def result_str(self, lr, period_time):
        self.compute_mean()
        str = f'Total Loss: {self.total_loss:.6f}\tLearning rate: {lr:.7f}\tTime: {period_time:.4f}\t'
        str += f'Diffusion Loss: {self.diff_loss:.6f}\tTriplet Loss: {self.triplet_loss:.6f}'
        return str