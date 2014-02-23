"""
For each IP compute the percentage of consecutive requests

AUTHORS::

    - Bill (bill@equalit.ie) 2012: Initial version
    - Vmon (vmon@equalit.ie) 2012: Took the average
    - Tomato (cerasiforme@gmail.com) 2013: Find percentage consecutive requests

"""
from learn2ban_feature import Learn2BanFeature
import operator
class FeaturePercentageConsecutiveRequests(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)

        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 10


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each
        ip. This feature returns the average depth of uri requests by IP.
        This is intended to distinguish between bots and real live persons.
        """
        #Vmon: Again why average has any meaning here, while normalizing will
        #take care care of it
        #Beside: I feel that wasn't the original intent, the original
        #intent is that in a website how many time you need to click to
        #reach that page which is not obtainable form the logs
        ip_recs = self._ip_sieve.ordered_records()



        for cur_ip_rec in ip_recs:
            last_req_folder = ""
            num_consec_reqs = 0
            no_html_requests = 0
            for payload in ip_recs[cur_ip_rec]:
                if payload.get_doc_type() == 'html':
                    cur_req_folder = payload.get_requested_element().rfind('/')
                    if cur_req_folder == last_req_folder:
                        num_consec_reqs +=1

                    no_html_requests += 1
                    last_req_folder = cur_req_folder


            self.append_feature(cur_ip_rec, no_html_requests and num_consec_reqs/float(no_html_requests) or 0)


        
