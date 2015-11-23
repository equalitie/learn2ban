"""
Unit tests for features

AUTHORS:

- Vmon (vmon@equalit.ie) 2012: initial version, unit tests for average, variance
"""

import unittest
from os.path import dirname
from os import getcwd, chdir
import sys
import cProfile
import datetime


try:
    src_dir  = dirname(dirname(__file__))
    if not src_dir: src_dir = "/home/vmon/doc/code/deflect/learn2ban/src"
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

#adding the src dir to the path for importing
sys.path.append(src_dir)

from ip_sieve import IPSieve
from features.src.learn2ban_feature import Learn2BanFeature
from features.src.feature_average_request_interval import FeatureAverageRequestInterval
from features.src.feature_variance_request_interval import FeatureVarianceRequestInterval
from features.src.feature_cycling_user_agent import FeatureCyclingUserAgent
from features.src.feature_html_to_image_ratio import FeatureHtmlToImageRatio
from features.src.feature_request_depth import FeatureRequestDepth
from features.src.feature_HTTP_response_code_rate import FeatureHTTPResponseCodeRate
from features.src.feature_payload_size_average import FeaturePayloadSizeAverage
from features.src.feature_request_depth_std import FeatureRequestDepthStd

class KnownValues(unittest.TestCase):
    pass

class BasicTests(unittest.TestCase):
    log_files = (src_dir+"/test/deflect_test.log", src_dir+"/test/deflect.log_cool1.20120810_five_percent.log")#src_dir+"/test/deflect.log_cool1.20120810.23h59m50s-20120812.00h00m00s.old" )
    #log_files = (src_dir+"/tests/deflect_test.log", src_dir+"/tests/deflect_test.log")
    test_ip_sieve = IPSieve()
    test_ip_feature_db = {}
    def __init__(self):
        pass
    def test_ip_sieve_parse(self):
        for cur_log_file in self.log_files:
            self.test_ip_sieve.add_log_file(cur_log_file)
            self.test_ip_sieve.parse_log()

    def test_all_features(self):
        for cur_log_file in self.log_files:
            self.test_ip_sieve.add_log_file(cur_log_file)
            self.test_ip_sieve.parse_log()

            for CurrentFeatureType in Learn2BanFeature.__subclasses__():
                cur_feature_tester = CurrentFeatureType(self.test_ip_sieve, self.test_ip_feature_db)
                cur_feature_tester.compute()

        print self.test_ip_feature_db

    def run_tests(self):
        """
        needs a function to feed to the profiler
        """
        self.test_ip_sieve_parse();
        self.test_all_features();

def run_foo():
    my_tester = BasicTests();
    my_tester.run_tests()

def profile_features():

    utc_datetime = datetime.datetime.utcnow()
    utc_datetime.strftime("%Y-%m-%d-%H%MZ")
    profile_log = 'profile_features_'  + str(utc_datetime) +".prof"

    cProfile.run('run_foo()', profile_log)
    #subprocess.call(['/usr/lib/python2.7/cProfile.py', '-o', profile_log, '../test_features.py']);

if __name__ == "__main__":
    profile_features()
    #unittest.main()
