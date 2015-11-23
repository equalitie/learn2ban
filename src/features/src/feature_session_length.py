"""
For each IP compute the length of time between first and last request (i.e., length of the session)

AUTHORS::

    - Vmon (vmon@equalit.ie) 2012: Initial version (average time of request)
    - Tomato (cerasiforme@gmail.com) 2013: Make into length of session instead

"""
from learn2ban_feature import Learn2BanFeature

class FeatureSessionLength(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)

        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 9


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each
        ip. This is basically the time of the last request  - first.
        """
        ip_recs = self._ip_sieve.ordered_records()

        for cur_ip_rec in ip_recs:
            feature_value = (len(ip_recs[cur_ip_rec]) > 1) and (ip_recs[cur_ip_rec][-1].time_to_second() - ip_recs[cur_ip_rec][0].time_to_second()) or 0
            #DEBUG
            #print len(ip_recs[cur_ip_rec]), ip_recs[cur_ip_rec][-1].time_to_second(),ip_recs[cur_ip_rec][0].time_to_second()
            #print feature_value
            self.append_feature(cur_ip_rec, feature_value)
