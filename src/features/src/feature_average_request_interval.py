"""
For each IP compute the average time between two request

AUTHORS::

    - Vmon (vmon@equalit.ie) 2012: Initial version

"""
from learn2ban_feature import Learn2BanFeature

class FeatureAverageRequestInterval(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)

        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 1

    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each
        ip. This is basically the time of the last request  - first / no of requests.
        """
        ip_recs = self._ip_sieve.ordered_records()
        for cur_ip_rec in ip_recs:
            # print len(ip_recs[cur_ip_rec])
            # print ip_recs[cur_ip_rec][-1].time_to_second(), ip_recs[cur_ip_rec][0].time_to_second()
            # print (ip_recs[cur_ip_rec][-1].time_to_second() - ip_recs[cur_ip_rec][0].time_to_second())/(len(ip_recs[cur_ip_rec])-1.0)
            feature_value = (len(ip_recs[cur_ip_rec]) > 1) and (ip_recs[cur_ip_rec][-1].time_to_second() - ip_recs[cur_ip_rec][0].time_to_second())/(len(ip_recs[cur_ip_rec])-1.0) or self.MAX_IDEAL_SESSION_LENGTH #If there's only one request then what average time mean? It should be infinity instead fo zero
            # print feature_value
            self.append_feature(cur_ip_rec, feature_value)
