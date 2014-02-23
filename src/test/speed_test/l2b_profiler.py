"""
profiles and generates time sheet for different modules  of Learn2Ban

AUTHORS:
 Vmon March 2013 Initial version
"""
from os.path import dirname, abspath
from os import getcwd, chdir
import sys
import cProfile
import datetime
import subprocess
import pdb
pdb.set_trace()

try:
    src_dir  = dirname(dirname(dirname(abspath(__file__))))
    if not src_dir: src_dir = "/lu101/sahosse/doc/code/deflect/learn2ban/src"
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

#adding the src dir to the path for importing
sys.path.append(src_dir)
sys.path.append(src_dir)


#I need a way to run cProfile on modules instead of functions because
#all I need is to profile.
def profile_features():
    import test.test_features
    
    utc_datetime = datetime.datetime.utcnow()
    utc_datetime.strftime("%Y-%m-%d-%H%MZ")
    profile_log = 'profile_features_'  + str(utc_datetime) +".prof"

    cProfile.run('test.test_features.run_tests()', profile_log)
    #subprocess.call(['/usr/lib/python2.7/cProfile.py', '-o', profile_log, '../test_features.py']);

if __name__ == "__main__":
    #profile all modules
    #import pdb
    #pdb.set_trace()
    profile_features()
