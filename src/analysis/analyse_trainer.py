"""
Analyse efficacy of learn2ban SVM system

AUTHORS:

- Bill (bill@equalit.ie) 2013/02/09
"""
from multiprocessing import Process
from os.path import dirname, abspath
from os import getcwd, chdir
import sys

try:
    src_dir = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

#our test svm to be trained
from sklearn import svm
import numpy as np

#train to ban
from train2ban import Train2Ban

from tools.learn2bantools import Learn2BanTools

import logging
import datetime

import cProfile

from l2b_experiment import L2BExperiment


class Analyser():
    # This approach is obsulite as now the L2BExperiments themselves
    # run the experiments and that can be marked in db

    # #user will send one of these values to make tweak the analyser behavoir
    # #train begin will take the begining portion of the data for training
    # #train random will choose random rows of the sample set
    # TRAIN_BEGIN = 0
    # TRAIN_RANDOM = 1
    # def __init__(self, where_to_train = TRAIN_BEGIN, training_portion = 1):
    #     """
    #     Intitiate the behavoir of the analyzer. These parametrs should be
    #     also tweakable from database

    #     INPUT:

    #       - where_to_train: which part of the sample should be used for
    #                         training
    #       - training_protion: Between 0 - 1, tells the analyser how much
    #                           of the sample is for training and how much
    #                           for testing.
    #     """
    #     self._where_to_train = where_to_train
    #     self._training_portion = training_portion

    def profile(self, exp, filename):
        cProfile.runctx('self.run_experiments(exp)', {'exp': exp}, locals(), filename)

    def run_experiments(self, exp):
        l2btools = Learn2BanTools()
        l2btools.load_train2ban_config()
        utc_datetime = datetime.datetime.utcnow()
        utc_datetime.strftime("%Y-%m-%d-%H%MZ")
        analyse_log_file = l2btools.analyser_results_dir + 'analyse_' + str(utc_datetime)

        logging.basicConfig(filename=analyse_log_file, level=logging.INFO)
        logging.info('Begin learn 2 ban analysis for Experiment Id: ' + str(exp['id']))

        l2btools.set_training_log(exp['training_log'])
        self.analyse_trainer = Train2Ban(l2btools.construct_classifier(exp['kernel_type']))
        self.analyse_trainer.add_malicious_history_log_files(l2btools.load_training_logs())
        training_set = l2btools.gather_all_features()
        self.analyse_trainer.add_to_sample(training_set)
        self.analyse_trainer.normalise(exp['norm_mode'])
        #This step will also update teh regex filter file to point to the experiment file
        self.analyse_trainer.add_bad_regexes(l2btools.load_bad_filters_from_db(exp['regex_filter_id']))
        #Train for training data
        self.analyse_trainer.mark_and_train()
        #Predict for training data using constructed model
        self.analyse_trainer.predict(training_set)
        logging.info('Training errors: ' + str(self.analyse_trainer.model_errors()))

        ip_index, data, target = self.analyse_trainer.get_training_model()
        bad_ips = [ip_index[cur_target] for cur_target in range(0, len(target)) if target[cur_target]]
        #logging.info('Training Bad IPs identified: '+ bad_ips)
        #Load testing data
        l2btools.set_testing_log(exp['testing_log'])
        #Clear IP_Sieve
        l2btools.clear_data()
        #Predict for testing data using model constructed from training data
        self.analyse_trainer.predict(l2btools.gather_all_features())

        logging.info('Testing errors: ' + str(self.analyse_trainer.model_errors()))
        experiment_result = {}
        experiment_result['experiment_id'] = exp['id']
        experiment_result['result_file'] = analyse_log_file
        l2btools.save_experiment(experiment_result)

        #here we would like to interfere and have some randomization.
        #Maybe as a pramater, we tell analyser what portion of the
        #sample should be used for training and what portion for
        #verification. Also, another parameter would be if that portion
        #should be taken from the begining of the sample or randomly.

    def run_l2b_experiments(self, exp, train_portion):
        l2btools = Learn2BanTools()
        l2btools.load_train2ban_config()
        utc_datetime = datetime.datetime.utcnow()
        utc_datetime.strftime("%Y-%m-%d-%H%MZ")
        analyse_log_file = l2btools.analyser_results_dir + 'analyse_' + str(utc_datetime)

        logging.basicConfig(filename=analyse_log_file, level=logging.INFO)
        logging.info('Begin learn 2 ban analysis for Experiment Id: ' + str(exp['id']))

        l2btools.set_training_log(exp['training_log'])

        experiment_classifier = l2btools.construct_classifier(exp['kernel_type'])
        self.analyse_trainer = Train2Ban(experiment_classifier)
        self.analyse_trainer.add_malicious_history_log_files(l2btools.load_training_logs())
        training_set = l2btools.gather_all_features()
        self.analyse_trainer.add_to_sample(training_set)
        self.analyse_trainer.normalise(exp['norm_mode'])
        #This step will also update teh regex filter file to point to the experiment file
        self.analyse_trainer.add_bad_regexes(l2btools.load_bad_filters_from_db(exp['regex_filter_id']))
        #marking training data
        self.analyse_trainer.mark_bad_target()

        marked_training_set = self.analyse_trainer.get_training_set()
        train_selector, test_selector = l2btools.random_slicer(len(marked_training_set), train_portion)
        train_set = marked_training_set.get_training_subset(case_selector=train_selector)
        test_set = marked_training_set.get_training_subset(case_selector=test_selector)

        #initializes L2BEXperiment
        cur_experiment = L2BExperiment(train_set, test_set, experiment_classifier)
        #cur_experiment.train()

        #Predict for training data using constructed model
        #logging.info('Crossvalidation score: ' + str(cur_experiment.cross_validate_test()))

        #graph the result
        dim_reducers = ['PCA', 'Isomap']
        kernels = ['linear', 'rbf', 'poly']
        all_possible_choices = np.transpose(np.array([np.tile(dim_reducers, len(kernels)), np.repeat(kernels, len(dim_reducers))]))

        for cur_choice in all_possible_choices:
            cur_experiment.plot(dim_reduction_strategy=cur_choice[0], kernel=cur_choice[1])

if __name__ == "__main__":
    l2btools = Learn2BanTools()
    l2btools.load_train2ban_config()
    experiment_set = l2btools.retrieve_experiments()
    for exp in experiment_set:
        p = Process(target=Analyser().run_experiments(exp))
        p.start()


    #TODO: add support for multiple experiment files
    #TODO: output results in formatted log
    #TODO: plot graphs of results
