"""
For each IP compute the standard diviation of depth of page requests

AUTHORS::

    - Sofia 2012: Initial version.

"""
from learn2ban_feature import Learn2BanFeature
import operator
import numpy as np
class FeatureRequestDepthStd(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)

        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 8


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
            total_html_requests = 0
            depth_list = []
            for payload in ip_recs[cur_ip_rec]:
                if payload.get_doc_type() == 'html':
                    page_depth = payload.get_requested_element().count('/')
                    depth_list.append(page_depth)

            self.append_feature(cur_ip_rec, len(depth_list)==0 and -1 or np.std(depth_list))
