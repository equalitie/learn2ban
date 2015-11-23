"""
The parent class for all features used to distinguished an attack from a legitimate request

AUTHORS::

    - Vmon (vmon@equalit.ie) 2012: Initial version

"""
class Learn2BanFeature(object):
    """
    We need to get the data for IPSieve class and analyze it. IPSieve
    provide the Feature with a dictionary and the feature needs to return
    a dictionary of IPs and numerical value

    (This is new-style particularly to loop through its children using
    __subclass__)
    """
    MAX_IDEAL_SESSION_LENGTH = 1800 #seconds
    def __init__(self, ip_sieve, ip_feature_db):
        """
        Set the corresponding ip_sieve

        INPUT::
           ip_sieve: the IPSieve object to crunch the ATS log file
           ip_feature_db: the global db that store all features of
                          all ips
        """
        self._ip_sieve = ip_sieve
        self._ip_feature_db = ip_feature_db

        self._FEATURE_INDEX = -1 #This is an abstract class so no real feature

    def compute(self):
        """
        The feature should overload this function to implement the feautere
        computation. At the end the results should be stored
        in a dictionary with the format of IP:value where the value is a double
        or an integer value
        """
        pass

    def append_feature(self, inspected_ip, feature_value):
        """
        Just checks if the ip is in the database adds the feature to it
        otherwise make a new record for the ip in the ip dictioanry

        INPUT::
             ispected_ip: the ip whose record we want to manipulate
             feature_value: the value that we want to add as
             {_FEATURE_INDEX: feature_value} to the record
        """
        if inspected_ip in self._ip_feature_db:
            self._ip_feature_db[inspected_ip][self._FEATURE_INDEX] = feature_value
        else:
            self._ip_feature_db[inspected_ip] = {self._FEATURE_INDEX:feature_value}
