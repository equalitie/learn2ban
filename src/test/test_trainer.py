"""
Unit tests for Train2Ban

AUTHORS:

- Vmon (vmon@equalit.ie) 2012: initial version
"""

import unittest
import numpy as np

from os.path import dirname, abspath
from os import getcwd, chdir
import sys

try:
    src_dir  = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

#train to ban
from tools.learn2bantools import Learn2BanTools
from tools.training_set import TrainingSet
from train2ban import Train2Ban

class KnownValues(unittest.TestCase):
    pass

class BasicTests(unittest.TestCase):
    TEST_LOG_FILENAME = src_dir + "/test/training_1000hit.log"
    TEST_LOG_ID = 0
    TEST_REGEX = "^<HOST> .*Firefox/1\.0\.1"
    
    def setUp(self):
        """Call before every test case."""
        self.l2btools = Learn2BanTools()
        self.l2btools.load_train2ban_config()

        self.l2btools.retrieve_experiments()
        self.log_files = [[self.TEST_LOG_ID, self.TEST_LOG_FILENAME]]

        #we are testing trainin
        self.test_trainer = Train2Ban(self.l2btools.construct_svm_classifier())
        self.test_trainer._training_set = TrainingSet() #clean the training set
        self.test_trainer.add_to_sample(self.l2btools.gather_all_features([self.TEST_LOG_FILENAME]))

    def test_normalization(self):
        self.test_trainer.normalise('sparse')
        self.test_trainer.normalise('individual')

    def test_training(self):
        self.test_trainer.normalise('individual')
        #indicate bad ips
        self.test_trainer.add_malicious_history_log_files([[self.TEST_LOG_ID, self.TEST_LOG_FILENAME]])

        self.test_trainer.add_bad_regexes(self.TEST_LOG_ID, [self.TEST_REGEX])
        self.test_trainer.mark_and_train()

        ip_index, data, target = self.test_trainer.get_training_model()

        bad_ips = [ip_index[cur_target][0] for cur_target in range(0,len(target))  if target[cur_target]]

        print "Bad IPs:",bad_ips
        #test pickling of model
        import datetime
        utc_datetime = datetime.datetime.utcnow()
        utc_datetime.strftime("%Y-%m-%d-%H%MZ")
        filename = 'l2b_pickle_'+str(utc_datetime)
        result = self.test_trainer.save_model(filename)
        self.test_trainer.save_model(filename+".normal_svm_model", "normal_svm")
        result = self.test_trainer.load_model(filename)

        #sizzeling for the sake of david's little kind heart
        # import numpy as np
        # import pylab as pl

        # X = [cur_row[0] for  cur_row in data]
        # Y = [cur_row[4] for  cur_row in data]

        # Z = self.tester_svm.fit(zip(X,Y), target)
        # pl.figure(0)
        # pl.clf()
        # pl.scatter(X, Y, c=target, zorder=10, cmap=pl.cm.Paired)
        # pl.axis('tight')

        # x_min = min(X)
        # x_max = max(X)
        # y_min = min(Y)
        # y_max = max(Y)

        # XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        # Z = self.tester_svm.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # # Put the result into a color plot
        # Z = Z.reshape(XX.shape)
        # pl.pcolormesh(XX, YY, Z > 0, cmap=pl.cm.Paired)
        # pl.contour(XX, YY, Z, colors=['k', 'k', 'k'],
        #            linestyles=['--', '-', '--'],
        #            levels=[-.5, 0, .5])

        # pl.title("Training result")
        # pl.show()

    def test_subset_training(self):
        #we are testing the training
        scoring_svm = self.test_trainer._ban_classifier

        #indicate bad ips
        self.test_trainer.add_malicious_history_log_files(self.log_files)
        self.test_trainer.add_bad_regexes(self.TEST_LOG_ID, (self.TEST_REGEX,))
        #retrieve the training set
        self.test_trainer.mark_bad_target()

        marked_training_set = self.test_trainer.get_training_set()

        #now we break the set into two sets
        print len(marked_training_set)
        first_halver = np.array([i <= len(marked_training_set)/2 for i in range(0, len(marked_training_set))])

        first_half = marked_training_set.get_training_subset(case_selector = first_halver)

        second_halver = np.logical_not(first_halver)
        second_half = marked_training_set.get_training_subset(case_selector = second_halver)
        print len(second_half) ,len(first_half)

        assert(abs(len(second_half) - len(first_half)) <= 1)

        #predicting the second half
        self.test_trainer.set_training_set(first_half)
        self.test_trainer.train()

        print "Predicting second half using first half. Score: ", scoring_svm.score(second_half._ip_feature_array, second_half._target)

        #predicting the first half
        self.test_trainer.set_training_set(second_half)
        self.test_trainer.train()

        print "Predicting first half using second half half. Score: ", scoring_svm.score(first_half._ip_feature_array, first_half._target)

        #now choose a random subset of size %10 and train
        random_selector, test_selector = self.l2btools.random_slicer(len(marked_training_set), train_portion = 0.1)
        self.test_trainer.set_training_set(marked_training_set.get_training_subset(random_selector))

        test_part = marked_training_set.get_training_subset(case_selector = test_selector)
        print "Predicting %90 of data using %10 random cases. Score: ", scoring_svm.score(test_part._ip_feature_array, test_part._target)

        #TODO test feature subselection

if __name__ == "__main__":
    unittest.main()
