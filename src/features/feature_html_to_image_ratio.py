"""
For each IP compute the ratio of HTML requests to Image requests

AUTHORS::
    
    - Bill (bill@equalit.ie) 2012: Initial version

"""

from learn2ban_feature import Learn2BanFeature
import operator
class FeatureHtmlToImageRatio(Learn2BanFeature):
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Simply calls the parent constructor
        """
        Learn2BanFeature.__init__(self, ip_sieve, ip_feature_db)
        
        #Each feature need to have unique index as the field number
        #in ip_feature_db
        self._FEATURE_INDEX = 3


    def compute(self):
        """
        retrieve the ip dictionary and compute the average for each 
        ip. This feature computes the ratio of HTML to Image requests for a given session.
        """
        ip_recs = self._ip_sieve.ordered_records()
#	print ip_recs
        for cur_ip_rec in ip_recs:
            doc_type_request_map = {}
            for payload in ip_recs[cur_ip_rec]:
                cur_type = payload.get_doc_type()
                if len(cur_type):
                    if cur_type not in doc_type_request_map:
                        doc_type_request_map[cur_type] = 1
                    else:
                        doc_type_request_map[cur_type] += 1

            """
            Current version looks at ratio of Images to HTML requested by given IP
            An extension or evoltion would be to look at ratio of resources such as
            JS and CSS as well
            """
            feature_value = 0 if ( 'image' not in doc_type_request_map or 'html' not in doc_type_request_map ) else ( float( doc_type_request_map['image'] ) / float( doc_type_request_map['html'] ) )
            #feature_value = 0 if ( 'image' not in doc_type_request_map or 'html' not in doc_type_request_map) else ( float( doc_type_request_map['image'] ) / float( doc_type_request_map['html'] ) )
            self.append_feature(cur_ip_rec, feature_value)

