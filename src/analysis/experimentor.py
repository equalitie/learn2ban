"""
Analyse efficacy of learn2ban SVM system

AUTHORS:

- Bill (bill@equalit.ie) 2013/02/09
- Vmon: July 2013: Change log status. Trying to return back to OOP model
        after learn2bantool reconstruction disaster
- Vmon: Nov 2013: store the experiment model along with the experminet
                  results
"""
from multiprocessing import Process
from os.path import dirname, abspath
import os
import sys

try:
    src_dir = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

from sklearn import svm
import numpy as np
import logging
import datetime

#learn2ban classes:
from ip_sieve import IPSieve

#feature classes
from features.src.learn2ban_feature import Learn2BanFeature

#train to ban and other tools
from train2ban import Train2Ban
from tools.training_set import TrainingSet

from tools.learn2bantools import Learn2BanTools

from l2b_experiment import L2BExperiment

nb_training = 10
training_portions = [x / float(nb_training) for x in range(1, nb_training)]

class Experimentor():
    """
    There is need for two type of Experiment objests one that correspond
    to each experiment record in experiment table and one that correspond
    to each result record in experiment_result.

    That is becaues from one experiment you can run many other experiments
    with little change in paramters and we don't want to store all these
    in DB as the design (train/test protion for example).

    Hence InseminatorExperiment read the experiment from the db (Expriment type 1)
    and Generator the L2BExperiment (Experiment type 2)
    """
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
    def __init__(self, exp, l2btools):
        """
        store the exp config in self's attribute.
        """
        self.expr_dict = exp
        self.id = self.expr_dict['id']
        self.l2btools = l2btools

        self.ip_sieve = IPSieve()
        self.ip_feature_db = {}

        #Create classifier, currently only SVM supported
        #but trainer is agnostic of classifier used provided it supports fit and predict
        self.experiment_classifier = self.l2btools.construct_svm_classifier(self.expr_dict['kernel_type'])
        #Create classifier
        self.trainer = Train2Ban(self.experiment_classifier)
        #Setup base data set
        #the base filename we are going to associate to the result of this experiment
        utc_datetime = datetime.datetime.utcnow()
        utc_datetime.strftime("%Y-%m-%d-%H%MZ")
        self.base_analyse_log_file = self.l2btools.analyser_results_dir + 'base_analyse_' + str(utc_datetime)
        #this make more sense to happens in the constructor however,
        self._process_logs()
        self._mark_bots()

    def param_stochastifier(self):
        """
        Here we return a randomised set of parameters for the experiments.
        At present we choose between for normalisation(sparse,individual), dimension reduction(PCA,ISOMap, MD5) and training portion(scale from 0-1)
        """
        param_set = []
        return param_set

    def _process_logs(self):
        """
        get the log name from db and gathers all features

        INPUT:
            log_files: the logs that we went through it.
        """
        #this is not a oop way of retrieving the logs but I think we are
        #avoiding db access in other classes beside l2btools
        cur_experiment_logs = self.l2btools.retrieve_experiment_logs(self.id)

        #if there is no log associated to this experiment then there is nothing
        #to do
        if len(cur_experiment_logs) == 0:
            logging.info("Giving up on experiment %i with no training log"%self.expr_dict['id'])
            return

        #log id is needed to be send to the trainer so the the trainer
        #knows which regex is detecting the bots for which log
        self.trainer.add_malicious_history_log_files([(cur_log_info['log_id'], cur_log_info['file_name']) for cur_log_info in cur_experiment_logs])

        #extracitng the filenames
        #Get IP Features
        log_filenames = tuple(cur_log['file_name'] for cur_log in cur_experiment_logs)
        #At this stage it is only a peliminary list we might lose features
        #due to 0 variance
        self._active_feature_list = []
        #do a dry run on all features just to gather the indeces of all available
        #features
        for CurrentFeatureType in Learn2BanFeature.__subclasses__():
            cur_feature_tester = CurrentFeatureType(self.ip_sieve, self.ip_feature_db)
            self._active_feature_list.append(cur_feature_tester._FEATURE_INDEX)

        for cur_log_file in log_filenames: #in theory it might be more memory efficient
            #to crunch the logs one by one but python is quite disappointing in memory
            #management
            try:
                self.ip_sieve.add_log_file(cur_log_file)
                self.ip_sieve.parse_log()
            except IOError:
                print "Unable to read ", cur_log_file, "skipping..."

        for CurrentFeatureType in Learn2BanFeature.__subclasses__():
            cur_feature_tester = CurrentFeatureType(self.ip_sieve, self.ip_feature_db)
            logging.info("Computing feature %i..."%cur_feature_tester._FEATURE_INDEX)
            cur_feature_tester.compute()

            # we have memory problem here :(
            # import objgraph
            # objgraph.show_refs([self.ip_sieve._ordered_records], filename='ips-graph.png')

        del self.ip_sieve._ordered_records
        del self.ip_sieve

        #fuck python with not letting the memory released
        # import gc
        # gc.collect()
        # print gc.garbage()

        self.trainer.add_to_sample(self.ip_feature_db)

        #we store the non-normailized vectors in a json file
        jsonized_ip_feature_db = {}
        for k,v in self.ip_feature_db.items():
            jsonized_ip_feature_db[str(k)] = v
        import json
        with open(self.base_analyse_log_file+".prenormal_ip_feature_db.json", "w") as ip_feature_file:
            json.dump(jsonized_ip_feature_db, ip_feature_file)

        del self.ip_feature_db
        del jsonized_ip_feature_db

        #Normalise training set, normalisation should happen after all
        #sample is gathered
        self.trainer.normalise(self.expr_dict['norm_mode'])

    def _mark_bots(self):
        """
        Read the regexes correspond to this experience log and apply them to
        the trainer. this should be called after the logs has been processed.
        """
        #Add Faill2Ban filters
        filters_for_experiment = self.l2btools.load_bad_filters_from_db(self.id)
        for cur_filter in filters_for_experiment:
            self.trainer.add_bad_regexes(cur_filter['log_id'], (cur_filter['regex'],))
        #Use Fail2ban filters to identify and mark DDOS IPs in data set
        malicious_ips = self.trainer.mark_bad_target()

        with open(self.base_analyse_log_file+".malicious_ip_list", "w") as malicious_ip_file:
            malicious_ip_file.write(str(malicious_ips).strip('[]'))

    def _pca_importance_ananlysis(self, pca_model):
        """
        Retrieve the pca transformation and use the following formula to
        determine the importance of each feature:

        length(variance*|c_1j|/sqrt(sum(c1i_2^2)))

        INPUT:
           pca_model: (the transfarmation matrix in np array, importance of each
                      component) the output of L2BExperiment.PCA_transform_detail
        OUTPUT: an array containing the importance ratio of features based
                on above forumla
        """
        pca_transform_matrix = pca_model[0]
        pca_var_ratio = pca_model[1]

        #row_sums = pca_transform_matrix.sum(axis=1)
        #apparently pca transfomation is normalised along both access
        #anyway for some reason reshape(-1) doesn't work as transpose
        scaled_coeffs = pca_var_ratio.reshape(len(pca_var_ratio),1) * pca_transform_matrix

        return np.apply_along_axis(np.linalg.norm, 0 , scaled_coeffs)

    def run_l2b_experiment(self, train_portion, stochastic_params):
        """
        Run individual instance of given experiment
        """
        utc_datetime = datetime.datetime.utcnow()
        utc_datetime.strftime("%Y-%m-%d-%H%MZ")
        analyse_log_file = self.l2btools.analyser_results_dir + 'analyse_' + str(utc_datetime)
        logging.basicConfig(filename=analyse_log_file, level=logging.INFO)
        logging.info('Begin learn 2 ban analysis for Experiment Id: ' + str(self.expr_dict['id']))

        #Divide up data set into training and testing portions based on initial given value
        marked_training_set = self.trainer.get_training_set()

        #if no body is a bot then this is not a fruitful experiment
        if marked_training_set.no_culprit():
            logging.info("No bot detected, Giving up on experiment " + str(self.expr_dict['id']))
            return

        #here we need to check if we lost features or not due to normalisation
        #sparse normaliastion doesn't cut off feature
        if self.expr_dict['norm_mode']=='individual':
            dimension_reducer = [cur_feature_std != 0 for cur_feature_std in marked_training_set._normalisation_data[marked_training_set.SAMPLE_STD]]
            self._active_feature_list = [self._active_feature_list[red_plc[0]] for red_plc in enumerate(dimension_reducer) if red_plc[1]]

        active_features = str(self._active_feature_list).strip('[]')
        #TODO: Iterate with different slicing to get reliable result
        train_selector, test_selector = self.l2btools.random_slicer(len(marked_training_set), train_portion)
        train_set = marked_training_set.get_training_subset(case_selector=train_selector)
        test_set = marked_training_set.get_training_subset(case_selector=test_selector)
        #initializes L2BEXperiment
        cur_experiment = L2BExperiment(train_set, test_set, self.trainer)

        #TODO:mRMR and PCA are independent of slicing and should
        #     computed over the whole dataset
        # Get the mRMR
        mrmr = cur_experiment.get_mrmr()
        logging.info('mRMR score: ' + str(mrmr))

        # Get the PCA ratios as a string
        pca_ratios = str(self._pca_importance_ananlysis(cur_experiment.pca_transform_detail())).strip('[]')
        logging.info('PCA ratios: ' + pca_ratios)

        #Train model against training set
        cur_experiment.train()

        #Predict for training data using constructed model
        score = cur_experiment.cross_validate_test()
        logging.info('Crossvalidation score: ' + str(score))

        self.store_results(analyse_log_file, train_portion, score, active_features, pca_ratios, mrmr)

    def store_results(self, analyse_log_file, train_portion, score, active_features, pca_ratios, mrmr):
        # Add the result to the database
        experiment_result = {}
        experiment_result['experiment_id'] = self.expr_dict['id']
        experiment_result['result_file'] = analyse_log_file
        experiment_result['proportion'] = train_portion
        experiment_result['score'] = score
        experiment_result['active_features'] = active_features
        experiment_result['pca_ratios'] = pca_ratios
        experiment_result['mrmr_score'] = str(mrmr).strip('[]')

        #while the pickle model is always created the result file only
        #get stored in the case there are an error
        self.l2btools.save_experiment_result(experiment_result)

        self.trainer.save_model(analyse_log_file+".l2b_pickle_model")
        #also try to store in recontsructable libsvm format if the function
        #if the save_svm_model function is implmented
        try:
            self.trainer.save_model(analyse_log_file+".normal_svm_model", "normal_svm")
        except NotImplementedError:
            print "save_svm_model is not implmeneted in your scikit-learn, skipping storing the model in libsvm format"

        print "Experiment", self.expr_dict['id'], ": train portion = ", train_portion, ", score = ", score, ", mRMR = ", mrmr, ", PCA ratios = ", pca_ratios
        print experiment_result

        # Graph the result
        # print cur_experiment.dim_reduction_PCA()
        # dim_reducers = ['PCA', 'Isomap']
        # kernels = ['linear', 'rbf', 'poly']
        # all_possible_choices = np.transpose(np.array([np.tile(dim_reducers, len(kernels)), np.repeat(kernels, len(dim_reducers))]))
        # for cur_choice in all_possible_choices:
        #     cur_experiment.plot(dim_reduction_strategy=cur_choice[0], kernel=cur_choice[1])

if __name__ == "__main__":
    l2btools = Learn2BanTools()
    l2btools.load_train2ban_config()

    desired_portions = [portion/100.0 for portion in range(10,100,10)]
    # Delete all previous experiments (will have to be commented out when performing series of experiments)
    #l2btools.delete_all_experiments_results()
    # Retrieve all the experiments stored in the db
    experiment_set = l2btools.retrieve_experiments()

    for exp in experiment_set:
        cur_expr = Experimentor(exp, l2btools)

        for cur_portion in desired_portions:
            stochastic_params = cur_expr.param_stochastifier()

            #single process
            #cur_expr.run_l2b_experiment(0.8, stochastic_params)

            #multi-processing: disabled temp cause hard to debug
            p = Process(target=cur_expr.run_l2b_experiment, args=(cur_portion, stochastic_params))
            p.start()
            print "process started with training prottion: ", cur_portion
            p.join()

        #objgraph.show_refs([cur_expr], filename='sample-graph.png')            #I guess this is a fall back if process can not start?
            #if pid == 0:
            #    print 'begin'
            #    Experimentor().run_l2b_experiment(exp, 0.8, l2btools, stochastic_params)

    # Display all the results
    print "all processes started"
    # experiment_results = l2btools.retrieve_experiments_results()
    # for res in experiment_results:
    #     print res

    #os._exit(0)

    #TODO: add support for multiple experiment files
    #TODO: output results in formatted log
    #TODO: plot graphs of results
