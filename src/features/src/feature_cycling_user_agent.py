"""
For each IP compute the average time between two request

AUTHORS::
    
    - Bill (bill@equalit.ie) 2012: Initial version

"""

from learn2ban_feature import Learn2BanFeature
import operator
class FeatureCyclingUserAgent(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)
        
        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 2


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each 
        ip to determine the change rate of UA per IP.
        """
        ip_recs = self._ip_sieve.ordered_records()

        for cur_ip_rec in ip_recs:
            ua_request_map = {}
            total_requests = 0
            highest_percentage_UA = 0;
            for payload in ip_recs[cur_ip_rec]:
                cur_UA = payload.get_UA()
                if cur_UA not in ua_request_map:                    
                    ua_request_map[cur_UA] = 1
                else:
                    ua_request_map[cur_UA] += 1
            
            #Sort UAs by number of requests
            sorted_ua_request_map = sorted(ua_request_map.iteritems(), key=operator.itemgetter(1), reverse=True)
            #Percentage of times UA has changed over the course of the requests
            feature_value = float(sorted_ua_request_map[0][1])/float(len(ip_recs[cur_ip_rec]))

            self.append_feature(cur_ip_rec, feature_value)
