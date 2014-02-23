"""
Unit tests for digesting the ATS records

AUTHORS:

- Vmon (vmon@equalit.ie) 2012: initial version
"""
import unittest
from os.path import dirname
from os import getcwd, chdir
import sys

try:
    src_dir  = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

#adding the src dir to the path for importing

from ip_sieve import IPSieve, ATSRecord
from tools.apache_log_muncher import parse_line as parse_apache_line

class KnownValues(unittest.TestCase):
    known_values = (('0.0.0.0 - [28/Sep/2012:16:14:01 -0800] "GET /assets/ico/hr-63567a98ba59eebda09903edeec1ff93.gif HTTP/1.1" http www.kavkaz-uzel.ru 304 0 "Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0" TCP_IMS_MISS - www.kavkaz-uzel.ru 319 http://www.kavkaz-uzel.ru/assets/ico/hr-63567a98ba59eebda09903edeec1ff93.gif', 1348863241),)

    def test_correct_seconds(self):
        for cur_value in self.known_values:
            cur_rec_dict = parse_apache_line(cur_value[0])
            test_record = ATSRecord(cur_rec_dict)
            assert(cur_value[1] == test_record.time_to_second())

# class BasicTests(unittest.TestCase):
#     log_files = ("deflect_test.log",)
#     test_ip_sieve = IPSieve()
#     test_ip_feature_db = {}

#     def test_ip_sieve_parse(self):
#         set_trace()
#         for cur_log_file in self.log_files:
#             self.test_ip_sieve.set_log_file(cur_log_file)
#             self.test_ip_sieve.parse_log()

if __name__ == "__main__":
    unittest.main()
