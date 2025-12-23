from aawedha.evaluation.evaluation_utils import aggregate_results
import numpy as np


class Score(object):
	def __init__(self):
		self.results = {}

	def build(self, metrics):
		"""initialize results as a dict with empty lists

		Parameters
		----------
		metrics : list
			metrics names
		"""
		self.results = {metric: [] for metric in metrics}
		self.results['probs'] = []
		self.results['confusion'] = []

	def update(self, results):
		"""Update metrics by append values to each metric list

		Parameters
		----------
		results : dict
			metrics and their values
		"""
		for metric in results:
			self.results[metric].append(results[metric])

	def results_reports(self, eval_results, classes, operations):
		"""Finalize metric calculation

		Parameters
		----------
		eval_results : dict
			evluation metric results
		classes : int
			class labels count
		operations : dict
			evaluation type and the operation indexes evaluated.
		"""
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
			if metric == 'probs' or metric == 'confusion':
				continue
			res[metric] = np.array(res[metric])
			res[metric + '_mean'] = res[metric].mean()
			res[metric + '_mean_per_fold'] = res[metric].mean(axis=0)
			if np.array(res[metric]).ndim == 2:
				res[metric + '_mean_per_subj'] = res[metric].mean(axis=1)
		res.update(operations)
		return res