"""
Parses a log file on the server and return the records corresponding to each client separately

AUTHORS:

 - Vmon (vmon@equalit.ie) 2012: Initial version.
 - Bill (bill.doran@gmail.com) 2012: lexify and other ATSRecord method depending on it.

"""

from time import strptime, mktime
import shlex

class ATSRecord:
    """
    This is to keep the info from one ATS record. For now we only extract
    the time but this can be change.

    TODO::
    We probably shouldn't read the whole table. There should be a way to
    temporally restrict the ispected data 
    """
    ATS_TIME_FORMAT = '%d/%b/%Y:%H:%M:%S'
    ATS_NO_FIELDS = 8 #maximum field index + 1 of the tokenized payload being 
                      #used in the feauter computation
    def __init__(self, ip, time, payload):
        self.ip = ip
        self.time = time
        self.payload = payload
        
        #run lexify once and for all to validate tokenised_payload
        self.lexify()

    def lexify(self):
        """
        Stores tockenize version of  the payload in a array
        """
	try: 
            self.tokenised_payload = shlex.split(self.payload, posix=False)
            #The posix=False will help with ignoring single single quotes
            #Other soltions:
            #1. getting rid of '
            #parsed_string.replace('\'','')

            #2. Use the shlex.shlex instead and tell it to ignore ' 
            # lex = shlex.shlex(str(self.payload))
            # lex.quotes = '"'
            # lex.whitespace_split = '.'
            # tokenised_payload = list(lex)
	    #return '' if len(tokenised_payload) <= 0 else tokenised_payload[payloadIndex]
            if len(self.tokenised_payload) <= 0:
                self.tokenized_payload = [''] * ATS_NO_FIELDS
	except ValueError, err:
            print(str(err))
            #for debug purpose
            print self.payload
            return '' #Return empty in case of error maintainig normal
                      #behavoir so the program doesn't crash
    def get_UA(self):
	"""
	Return the User Agent for this payload
	"""
        return self.tokenised_payload[5]

    def time_to_second(self):
        """
        convert the time value to total no of seconds passed
        since ???? to facilitate computation.
        """
        #find to ignore time-zone
        digested_time = strptime(self.time[:self.time.find(' ')], self.ATS_TIME_FORMAT)
        return mktime(digested_time)
    def get_doc_type(self):
	"""
	Retrieves the document type, if present, for the current payload
	"""
        return self.tokenised_payload[7]
    def get_payload_size(self):
	"""
	Retrieves the payload size, if present, for the current payload
	"""

        return self.tokenised_payload[4]
    def get_http_status_code(self):
	"""
	Retrieves the HTTP status code, if present, for the current payload
	"""

        return self.tokenised_payload[3]
    def get_requested_element(self):
	"""
	Retrieves the requested uri, if present, for the current payload
	"""

        return self.tokenised_payload[0]

class IPSieve():
    def __init__(self, log_filename=None):
        self._ordered_records = {}

        #This tells the sieve that needs to re-read data from the file.
        if (log_filename):
            set_log_file(self, log_filename)
        else:
            #If no file is specied then no record means all records
            self.dict_invalid = False
            self._log_filename = None
            self._log_lines = None #can be a file handle or array of lines

    def set_log_file(self, log_filename):
        """
        It takes the name of the log file and open the file
        throw and exception if not successful.
        """
        self.dict_invalid = True
        self._log_filename = log_filename
        self._log_lines = open(self._log_filename)


    def set_log_lines(self, log_lines):
        """
        It takes an array of log lines
        """
        self.dict_invalid = True
        self._log_lines = log_lines

    def parse_log(self):
        """
        Read each line of the log file and batch the
        records corresponding to each client (ip) 
        make a dictionary of lists each consisting of all records
        """
        for cur_rec in self._log_lines:
            #Here (at least for now) we only care about the ip and the time record.
            time_pos = cur_rec.find('-')
            if time_pos == -1: #Not a valid record
                continue

            http_req_pos = cur_rec.find('"')
            cur_ip = cur_rec[:time_pos-1]
            rec_time = cur_rec[time_pos + 3:http_req_pos - 2]
            rec_payload = cur_rec[http_req_pos:]
            #check if we have already encountered this ip
            cur_ats_rec = ATSRecord(cur_ip, rec_time, rec_payload)
            if not cur_ip in self._ordered_records:
                self._ordered_records[cur_ip] = [cur_ats_rec]
            else:
                self._ordered_records[cur_ip].append(cur_ats_rec)

        self.dict_invalid = False

    def ordered_records(self):
        """
        Wrapper for the record dictionary
        """
        if (self.dict_invalid):
            self.parse_log()

        return self._ordered_records
    
