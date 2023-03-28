from collections import defaultdict

import numpy as np

from concurrent.futures import ThreadPoolExecutor



class ComputeMetrics():

    def __init__(self, q, match_points, outfile='./metrics'):
        self.q = q
        self.match_points = sorted(list(set(match_points)))
        self.max_match_point = self.match_points[-1]
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.count = 0
        self.out = open(outfile, 'a')
        self.metrics = defaultdict(lambda: defaultdict(list))

    def compute_metrics(self):
        while True:
            I, iid, finish_flag, end_flag = self.q.get()
            if end_flag:
                break
            if not finish_flag:
                self.add_one_batch(I, iid)
            else:
                self.print_metrics()

    def reset(self):
        self.count += 1
        self.out.write('=' * 52 + f'  test epoch-{self.count:4d} finished  ' + '=' * 52 + '\n')
        self.out.write('#' * 132 + '\n')
        self.out.flush()
        self.metrics = defaultdict(lambda: defaultdict(list))

    @staticmethod
    def _compute_metrics(I, iid, hits_matrix, ndcgs, mrrs, per):
        per_pred = I[per]
        per_hits_matrix = hits_matrix[per]
        per_ndcgs = ndcgs[per]
        per_mrrs = mrrs[per]
        per_iid = np.squeeze(iid[per])
        for pred_point, perd in enumerate(per_pred):
            if perd == per_iid:
                per_hits_matrix[pred_point:] = 1
                per_ndcgs[pred_point:] = 1.0 / (np.log2(pred_point + 2))
                per_mrrs[pred_point:] = 1.0 / (pred_point + 1)
                break

    def add_one_batch(self, I, iid):
        batch_size = len(I)
        match_points = self.match_points
        max_match_point = self.max_match_point

        hits_matrix = np.zeros(shape=(batch_size, max_match_point), dtype=np.int64)
        ndcgs = np.zeros(shape=(batch_size, max_match_point))
        mrrs = np.zeros(shape=(batch_size, max_match_point))

        tasks = [
            self.executor.submit(self._compute_metrics, I, iid, hits_matrix, ndcgs, mrrs, per) \
            for per in range(batch_size)
        ]
        for task in tasks:
            task.result()

        batch_metrics = defaultdict(dict)
        for match_point in match_points:
            batch_metrics['HR'][match_point] = hits_matrix[:, match_point - 1]
            batch_metrics['NDCG'][match_point] = ndcgs[:, match_point - 1]
            batch_metrics['MRR'][match_point] = mrrs[:, match_point - 1]

        for per_metric, per_batch_metric_values in batch_metrics.items():
            for match_point, per_batch_metric_value in per_batch_metric_values.items():
                self.metrics[per_metric][match_point].append(per_batch_metric_value)

    def print_metrics(self):
        aggregated_metrics = defaultdict(dict)
        for per_metric, per_metric_values in self.metrics.items():
            for math_point, per_metric_value in per_metric_values.items():
                aggregated_metrics[per_metric][math_point] = \
                    np.mean(np.concatenate(per_metric_value, axis=0))
        print('\n' + '#' * 132)
        for per_metric, per_aggregated_metric_values in aggregated_metrics.items():
            header = '=' * 52 + f'    {per_metric}    ' + '=' * 52
            print(header)
            self.out.write(header + '\n')
            per_metric_str = ''
            for math_point, per_aggregated_metric_value in per_aggregated_metric_values.items():
                per_aggregated_metric_value *= 100
                per_metric_str = per_metric_str + '-' + \
                    f' @{math_point:3d}: {per_aggregated_metric_value:.4f}% ' + '-'
                if len(per_metric_str) > 120:
                    print(per_metric_str)
                    self.out.write(per_metric_str + '\n')
                    per_metric_str = ''
            if per_metric_str:
                print(per_metric_str)
                self.out.write(per_metric_str + '\n')
        self.out.flush()
        self.reset()
