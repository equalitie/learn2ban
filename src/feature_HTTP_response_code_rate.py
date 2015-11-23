"""
For each IP compute the HTTP error rate for a given set of requests

AUTHORS::
    
    - Bill (bill@equalit.ie) 2012: Initial version

"""
from learn2ban_feature import Learn2BanFeature
import operator
class FeatureHTTPResponseCodeRate(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)
        
        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 6


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each 
        ip. This feature returns the rate of HTTP error codes for a given IP address.
        """
        ip_recs = self._ip_sieve.ordered_records()

        for cur_ip_rec in ip_recs:
		total_error_statuses = 0
		total_requests = 0
		for payload in ip_recs[cur_ip_rec]:	
			status_code = int(payload.get_http_status_code() or 0)
			if status_code >= 400 and status_code < 500:
				total_error_statuses += 1
			total_requests += 1
		#Percentage of http status errors over the course of the requests
		feature_value = 0 if total_error_statuses <= 0 else float(total_error_statuses)/float(total_requests)

                self.append_feature(cur_ip_rec, feature_value)
			

