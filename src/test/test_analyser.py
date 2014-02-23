"""
Unit tests for Analyser

AUTHORS:

- Bill (bill@equalit.ie) 2013: initial version
"""
from multiprocessing import Process

import unittest
from os.path import dirname, abspath
from os import getcwd, chdir
import sys

try:
    src_dir  = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

from analysis.experimentor import Experimentor
from tools.learn2bantools import Learn2BanTools

class BasicTest(unittest.TestCase):
    l2btools = Learn2BanTools()
    experiment_set = ({'regex_filter_id': 1L, 'testing_log': 'testing.log', 'kernel_type': 'linear', 'training_log': 'training.log', 'norm_mode': 'individual', 'id': 2L},)

    def nontest_analyser(self):
        self.l2btools.load_train2ban_config()
	experiment_set = self.l2btools.retrieve_experiments()
        for exp in experiment_set:
            p = Process(target=Analyser().run_experiments(exp))
            p.start()

    def test_l2b_experiment(self):
        self.l2btools.load_train2ban_config()
        for exp in self.experiment_set:
            cur_experimentor = Experimentor(exp, self.l2btools)
            cur_experimentor.run_l2b_experiment(0.70, [])

if __name__ == "__main__":
    unittest.main()
