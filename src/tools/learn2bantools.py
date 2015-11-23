"""
Utility class that holds commonly used train 2 ban functions

AUTHORS:

- Bill (bill@equalit.ie) 2013/02/13
"""

from ip_sieve import IPSieve
from train2ban import Train2Ban
from tools.training_set import TrainingSet

#feature classes
from features.src.learn2ban_feature import Learn2BanFeature
from features.src.feature_average_request_interval import FeatureAverageRequestInterval
from features.src.feature_cycling_user_agent import FeatureCyclingUserAgent
from features.src.feature_html_to_image_ratio import FeatureHtmlToImageRatio
from features.src.feature_variance_request_interval import FeatureVarianceRequestInterval
from features.src.feature_HTTP_response_code_rate import FeatureHTTPResponseCodeRate
from features.src.feature_payload_size_average import FeaturePayloadSizeAverage
from features.src.feature_request_depth import FeatureRequestDepth
from features.src.feature_request_depth_std import FeatureRequestDepthStd
from features.src.feature_session_length import FeatureSessionLength
from features.src.feature_percentage_consecutive_requests import FeaturePercentageConsecutiveRequests

import math
import ConfigParser
import numpy as np
from sklearn import svm

from os.path import dirname, abspath
from os import getcwd, chdir

import sys
import glob
import xml.etree.ElementTree as ET

import MySQLdb

try:
    src_dir = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

class Learn2BanTools():
    ip_sieve = IPSieve()
    ip_feature_db = {}
    log_files = list()

    def load_data_logs(self):
        """
        Retrieve all training logs from testing log directory
        """
        data_log_list = glob.glob(self.data_dir + '*')
        self.log_files = data_log_list
        return self.log_files

    def connect_to_db(self):
        """
        This connetcion to the db will live for the live time of the
        learn2bantools instance and will be used to save data back to the db
        """
        self.db = MySQLdb.connect(self.db_host, self.db_user, self.db_password)

        #Create cursor object to allow query execution
        self.cur = self.db.cursor(MySQLdb.cursors.DictCursor)
        sql = 'CREATE DATABASE IF NOT EXISTS learn2ban'
        self.cur.execute(sql)

	#Connect directly to DB
        self.db = MySQLdb.connect(self.db_host, self.db_user, self.db_password, self.db_name)
        self.cur = self.db.cursor(MySQLdb.cursors.DictCursor)

    def disconnect_from_db(self):
        """
        Close connection to the database
        """
        self.cur.close()
        self.db.close()

    # def save_experiment(self, experiment_result):
    #     """
    #     Save the results of an experimental run (old version)
    #     """
    #     add_experiment_result = ("INSERT INTO experiment_result( experiment_id, result_file) VALUES (%(experiment_id)s,%(result_file)s)")
    #     self.cur.execute(add_experiment_result, experiment_result)
    #     self.db.commit()

    def save_experiment_result(self, experiment_result):
        """
        Saves the result of an experimental run, including testing proportion used and score
        """
        add_experiment_result = ("INSERT INTO experiment_results( experiment_id, result_file, proportion, score, active_features, pca_ratios, mrmr_score) VALUES (%(experiment_id)s,%(result_file)s,%(proportion)s,%(score)s,%(active_features)s,%(pca_ratios)s,%(mrmr_score)s)")
        self.cur.execute(add_experiment_result, experiment_result)
        self.db.commit()

    def retrieve_experiments_results(self):
        """
        Retrieve the results of the experiments already run
        """
        self.cur.execute("SELECT * FROM experiment_results")
        experiment_results = self.cur.fetchall()
        return experiment_results

    def delete_all_experiments_results(self):
        """
        Drops the entire experiment_results table
        """
        self.cur.execute("TRUNCATE TABLE experiment_results")

    def retrieve_experiments(self):
        """
        Retrieve the set of experiments to run from the database
        """
        self.cur.execute("SELECT * FROM experiments where enabled=TRUE")
        self.experiment_set = self.cur.fetchall()
        return self.experiment_set

    def retrieve_experiment_logs(self, experiment_id):
        """
        Read the experiment_logs table and retrieve the name
        of logs associated to the experiment id

        INPUT:
           experiment_id: the id of the experiment whose logs are sought
        """
        self.cur.execute("SELECT experiment_logs.log_id, logs.file_name FROM experiment_logs, logs WHERE experiment_logs.log_id = logs.id AND experiment_logs.experiment_id =" + str(experiment_id) + ";")
        log_set =  self.cur.fetchall()
        #add the full path to log files
        for cur_log in log_set:
            cur_log['file_name'] = self.data_dir + cur_log['file_name']
        return log_set

    def load_database_config(self):
        """
        Get configuration parameters from the learn2ban config file
        and from the lern2ban database
        """
        config = ConfigParser.ConfigParser()
        config.readfp(open(src_dir+'/config/train2ban.cfg'))
        self.db_user = config.get('db_params', 'db_user')
        self.db_password = config.get('db_params', 'db_password')
        self.db_host = config.get('db_params', 'db_host')
        self.db_name = config.get('db_params', 'db_name')
        self.config_profile = config.get('db_params', 'config_profile')

    def load_train2ban_config(self):
        #Get database connection params
        self.load_database_config()
        #Establish database connection object
        self.connect_to_db()
        #Get basic config parameters
        #first try to see if there is a config specific to this host
        if (not self.config_profile):
            self.config_profile = "default"

        self.cur.execute("SELECT * from config where profile_name='"+ self.config_profile+"';")
        config_row = self.cur.fetchone()

        #otherwise we read first row of the database (the one with minimum id)
        if (not config_row):
            self.cur.execute("SELECT * from config ORDER BY id ASC")
            config_row = self.cur.fetchone()

        if (not config_row):
            raise IOError, "No configuration record in the database"

        if not config_row["absolute_paths"]:
            cur_dir = src_dir
        else:
            cur_dir = ""

        try:
            self.data_dir = cur_dir + config_row["training_directory"] + (config_row["training_directory"][-1] != "/" and "/" or "")
            self.analyser_results_dir = cur_dir + config_row["analyser_results_directory"] +  (config_row["analyser_results_directory"][-1] != "/" and "/" or "")

        except IndexError:
            raise ValueError, "Data and Result directory can not be left blank"

        #depricated for now, we are entering the regexes directly into the db
        if config_row["regex_filter_directory"]:
            self.filter_dir = cur_dir + config_row["regex_filter_directory"]  + (config_row["regex_filter_directory"][-1] != "/" and "/" or "")
            
        if config_row["default_filter_file"]:
            self.filter_file = self.filter_dir + config_row["default_filter_file"] +  (config_row["default_filter_file"][-1] != "/" and "/" or "")

    def add_data_log(self, log):
        self.log_files = list()
        self.log_files.append(self.train_dir + log)

    def load_bad_filters_from_db(self, experiment_id):
        #TODO: ensure cur is live
        self.cur.execute("SELECT regex_assignment.log_id, regex_filters.regex from regex_assignment, regex_filters, experiment_logs WHERE regex_assignment.regex_filter_id = regex_filters.id AND regex_assignment.log_id = experiment_logs.log_id AND experiment_logs.experiment_id = " +str(experiment_id))
        return self.cur.fetchall()

    def load_bad_filters(self):
        """
        Load set of regex filters from the default filter file.
        This is to allow expression of an individual filter file, rather
        than a set or by experiment.
        """
        tree = ET.parse(self.filter_file)
        root = tree.getroot()
        filters = list()
        for child in root:
            filters.append(child.text)
        return filters

    def sieve_the_ip(self):
        """
        This was used when all experiment were using all of log files
        but in new model each experiment has its own file
        """
        for cur_log_file in self.log_files:
            self.ip_sieve.add_log_file(cur_log_file)
            self.ip_sieve.parse_log_file()

    # def magnitude(self,v):
    #     return math.sqrt(sum(v[i]*v[i] for i in v))

    def clear_data(self):
        self.ip_sieve = IPSieve()
        self.ip_feature_db = {}

    def gather_all_features(self, log_files):
        """
        gathers all features

        INPUT:
            log_files: the logs that we went through it.
        """
        for cur_log_file in log_files:
            self.ip_sieve.add_log_file(cur_log_file)
            self.ip_sieve.parse_log()
            for CurrentFeatureType in Learn2BanFeature.__subclasses__():
                cur_feature_tester = CurrentFeatureType(self.ip_sieve, self.ip_feature_db)
                cur_feature_tester.compute()

        return self.ip_feature_db

    def construct_svm_classifier(self, kernel_mode='linear'):
        """
        Creates an instance of the SVM classifier with a given mode
        """
        return svm.SVC(kernel=kernel_mode)

    def random_slicer(self, data_size, train_portion=0.5):
        """
        Return two arrays with random true and false and complement of each
        other, used for slicing a set into trainig and testing

        INPUT:
            data_size: size of the array to return
            train_portion: between 0,1 indicate the portion for the True
                           entry
        """
        from random import random
        random_selector = [random() < train_portion for i in range(0, data_size)]
        complement_selector = np.logical_not(random_selector)

        return random_selector, complement_selector

    def __init__(self):
        #we would like people to able to use the tool object even
        #if they don't have a db so we have no reason to load this
        #config in the constructor
        #self.load_database_config()
        pass
