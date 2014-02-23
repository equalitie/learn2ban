"""
Profiler for Train2Ban

AUTHORS:

- Bill (bill@equalit.ie) 2013: initial version
"""

import cProfile
import pstats

import unittest
from os.path import dirname, abspath
from os import getcwd, chdir
import sys
import datetime

try:
    src_dir  = dirname(dirname(abspath(__file__)))
except NameError:
    #the best we can do to hope that we are in the test dir
    src_dir = dirname(getcwd())

sys.path.append(src_dir)

#train to ban
from analysis.analyse_trainer import Analyser
from tools.learn2bantools import Learn2BanTools
import pstats

class Profiler():
    l2btools = Learn2BanTools()
    def profile_learn2ban(self):
        self.l2btools.load_train2ban_config()
        experiment_set = self.l2btools.retrieve_experiments()
        for exp in experiment_set:
            utc_datetime = datetime.datetime.utcnow()
            utc_datetime.strftime("%Y-%m-%d-%H%MZ")
            filename = src_dir+'/profiler/logs/profile_'+str(utc_datetime)
            Analyser().profile(exp,0.8, filename)
            break
        p = pstats.Stats(filename)
        p.strip_dirs().sort_stats(-1).print_stats()
        p.dump_stats(src_dir+'profiler/logs/stats_'+str(utc_datetime))
#            cProfile.run(a.run_experiments(exp))#, 'profile'+str(utc_datetime))
if __name__ == "__main__":
    Profiler().profile_learn2ban() 
