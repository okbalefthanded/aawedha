from aawedha.evaluation.evaluation_utils import aggregate_results
import numpy as np


class Score(object):

	def __init__(self):
		self.results = {}

	def build(self, metrics):
		self.results = {metric: [] for metric in metrics}
		self.results['probs'] = []
		self.results['confusion'] = []

	def update(self, results):
		for metric in results:
			self.results[metric].append(results[metric])

	def results_reports(self, eval_results, classes, operations):
		eval_results = aggregate_results(eval_results)
		eval_results = self._update_results(eval_results, classes, operations)
		self.results = eval_results

	def _update_results(self, res, classes, operations):
		"""Add metrics results mean to results dict.

		Parameters
		----------
		res : dict
			dictionary of metrics results
		Returns
		-------
		res : dict
			updated dictionary with metrics mean fields.
		"""
		metrics = list(res.keys())

		if classes == 2:
			metrics.remove('viz')

		for metric in metrics:
			if metric == 'probs':
				continue
			res[metric] = np.array(res[metric])
			res[metric + '_mean'] = res[metric].mean()
			res[metric + '_mean_per_fold'] = res[metric].mean(axis=0)
			if np.array(res[metric]).ndim == 2:
				res[metric + '_mean_per_subj'] = res[metric].mean(axis=1)
		res.update(operations)
		return res