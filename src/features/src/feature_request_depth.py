"""
For each IP compute the average depth of page requests

AUTHORS::

    - Bill (bill@equalit.ie) 2012: Initial version
    - Vmon (vmon@equalit.ie) 2012: Took the average

"""
from learn2ban_feature import Learn2BanFeature
import operator
class FeatureRequestDepth(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)

        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 7


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
            total_page_depth = 0
            total_html_requests = 0
            for payload in ip_recs[cur_ip_rec]:
                if payload.get_doc_type() == 'html':
                    page_depth = payload.get_requested_element().count('/')
                    total_page_depth += page_depth
                    total_html_requests += 1

            self.append_feature(cur_ip_rec, total_html_requests and total_page_depth/total_html_requests or 0)
