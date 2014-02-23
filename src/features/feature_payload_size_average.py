"""
For each IP compute the average time between two request

AUTHORS::

    - Bill (bill@equalit.ie) 2012: Initial version

"""

from learn2ban_feature import Learn2BanFeature
import operator
class FeaturePayloadSizeAverage(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)

        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 5


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each
        ip to determine the change rate of UA per IP.
        """
        ip_recs = self._ip_sieve.ordered_records()

        #Vmon: obviously the total size comparative to all other sizes is important
        #so we are better off not to divide because the normalizer will
        #compute that value and apply it during prediction

        #Vmon: totally bullshit. Normalization divides by all request of all 
        #requesters while here we are only average for this specific ip
        for cur_ip_rec in ip_recs:
            total_size = 0
            for payload in ip_recs[cur_ip_rec]:
                total_size += int(payload.get_payload_size())

            #Calculate average pyalod size for given IP
            self.append_feature(cur_ip_rec, (total_size > 0) and total_size / len(ip_recs[cur_ip_rec]) or 0)
