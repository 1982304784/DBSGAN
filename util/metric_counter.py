from collections import defaultdict
import numpy as np
from tensorboardX import SummaryWriter

WINDOW_SIZE = 100


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)
        self.metrics = defaultdict(list)
        self.metric_name = ['PSNR', 'SSIM']
        self.best_metric = 0

    def clear(self):
        self.metrics = defaultdict(list)

    def add_losses(self, loss_dic):
        for name, value in loss_dic.items():
            self.metrics[name].append(value)

    def add_metrics(self, metrics):
        for name, value in metrics.items():
            self.metrics[name].append(value)

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ['PSNR', 'SSIM'])
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def metric_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in self.metric_name)
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in self.metrics.keys():
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)

    def update_best_model(self):
        cur_metric = np.mean(self.metrics['PSNR'])
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
