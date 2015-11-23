"""
For each IP compute the variance of the time intervals between two
requests. That is:

var(X) = sum (x_i - \bar{X})^2/(n-1)

This is divided by n-1 because we are actually estimating the stddiv of the
bot using a finite sample of the society.

AUTHORS::
    
    - Vmon (vmon@equalit.ie) 2012: Initial version
    - Vmon Nov 2013: Using numpy instead of manual std

"""
from learn2ban_feature import Learn2BanFeature
import numpy as np

class FeatureVarianceRequestInterval(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)
        
        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 4


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each 
        ip. Then walk between each two consecutive requests and compute the
        difference of their time lag and mean. Basically the pencil and
        paper way. 

        (a_i - a_{i -1} - mu)^2 +...+ (a_{i+1}- a_i - mu)^2

        TODO:: This can improved. In the way that we compute the moving 
        varianceinstead. However, this need the feature object to remember 
        the old calculation which is very reasonable requirement. We should
        move to that model soon.
        """
        ip_recs = self._ip_sieve.ordered_records()

        for cur_ip_rec in ip_recs:
            sample_size = len(ip_recs[cur_ip_rec]) - 1

            if sample_size < 2:
                #the variance of single value is 0 obviously
                feature_value = 0
                
            else:
                interval_list = []
                #just storing each interval in a list
                for i in xrange(0, sample_size):
                    cur_interval = ip_recs[cur_ip_rec][i+1].time_to_second()  - ip_recs[cur_ip_rec][i].time_to_second()
                    interval_list.append(cur_interval)

                feature_value  = np.std(interval_list)

            self.append_feature(cur_ip_rec, feature_value)










