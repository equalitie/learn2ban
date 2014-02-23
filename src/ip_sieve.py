"""
Parses a log file on the server and return the records corresponding to each client separately

AUTHORS:

 - Vmon (vmon@equalit.ie) 2012: Initial version.
 - Bill (bill.doran@gmail.com) 2012: lexify and other ATSRecord method depending on it.
 - Vmon Oct 2013: Session tracking added.

"""

from time import strptime, mktime
from tools.apache_log_muncher import parse_line as parse_apache_line

class ATSRecord:
    """
    This is to keep the info from one ATS record. For now we only extract
    the time but this can be change.

    INPUT:
        cur_rec_dict: a dictionary resulted from
    TODO::
    We probably shouldn't read the whole table. There should be a way to
    temporally restrict the inspected data
    """
    #ATS_TIME_FORMAT = '%d/%b/%Y:%H:%M:%S'
    ATS_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
    ATS_NO_FIELDS = 8 #maximum field index + 1 of the tokenized payload being
                      #used in the feauter computation
    #to decide that the session is dead and we need to start a new session

    def __init__(self, cur_rec_dict):
        self.ip = cur_rec_dict["host"]
        self.time = cur_rec_dict["time"];
        self.payload = cur_rec_dict;

        #do not run lexify it is slow
        #self.lexify()

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
        return self.payload["agent"]

    def time_to_second(self):
        """
        convert the time value to total no of seconds passed
        since ???? to facilitate computation.
        """
        #find to ignore time-zone
        try:
            digested_time = strptime(self.time[:self.time.find('Z')], self.ATS_TIME_FORMAT)
        except (ValueError):
            print "time is ", self.time

        return mktime(digested_time)

    def get_doc_type(self):
	"""
	Retrieves the document type, if present, for the current payload
	"""
        return self.payload["type"]

    def get_payload_size(self):
	"""
	Retrieves the payload size, if present, for the current payload
	"""
        return self.payload["size"]

    def get_http_status_code(self):
	"""
	Retrieves the HTTP status code, if present, for the current payload
	"""
        return self.payload["status"]

    def get_requested_element(self):
	"""
	Retrieves the requested uri, if present, for the current payload
	"""
        return self.payload["request"]

class IPSieve():
    DEAD_SESSION_PAUSE  = 1800 #minimum number of seconds between two session

    def __init__(self, log_filename=None):
        self._ordered_records = {}
        self._log_file_list = []

        #This tells the sieve that needs to re-read data from the file.
        if (log_filename):
            add_log_file(self, log_filename)
        else:
            #If no file is specied then no record means all records
            self.dict_invalid = False
            self._log_lines = None #can be a file handle or array of lines

    def add_log_file(self, log_filename):
        """
        It takes the name of the log file and store it in a list
        """
        self._log_file_list.append(log_filename)
        self.dict_invalid = True

    def add_log_files(self, log_filename_list):
        """
        It takes a list of name of the log files and extend the filelist to it
        """
        self._log_file_list.extend(log_filename_list)
        self.dict_invalid = True

    def set_log_lines(self, log_lines):
        """
        It takes an array of log lines
        """
        self.dict_invalid = True
        self._log_lines = log_lines

    def set_pre_seived_order_records(self, pre_seived_records):
        """
        It sets the order records directy to the dictionary
        supplied by the user
        """
        self.dict_invalid = False
        self._ordered_records = pre_seived_records

    def parse_log(self):
        """
        Read each line of the log file and batch the records corresponding
        to each client (ip) make a dictionary of lists each consisting of all
         records
        """
        #to check the performance and the sensitivity of the log mancher
        total_failure_munches = 0
        for log_filename in self._log_file_list:
            try:
                self._log_lines = open(log_filename)
            except IOError:
                raise IOError

            self._log_lines.seek(0, 2) #go to end to check the size
            total_file_size = self._log_lines.tell()
            self._log_lines.seek(0, 0) #and go back to the begining
            previous_progress = 0

            print "Parsing ", log_filename.split('/')[-1]

            #we are going to keep track of each ip and last session number corresponding
            #to that ip
            ip_session_tracker = {}
            for cur_rec in self._log_lines:
                new_session = False
                cur_rec_dict = parse_apache_line(cur_rec)

                if cur_rec_dict:
                    cur_ip = cur_rec_dict["host"];
                    cur_ats_rec = ATSRecord(cur_rec_dict);

                    if not cur_ip in ip_session_tracker:
                        ip_session_tracker[cur_ip] = 0
                        new_session = True

                    #now we are checking if we hit a new session
                    #if we already decided that we are in a new session then there is nothing
                    #to investigate
                    if not new_session:
                        #so we have a session already recorded, compare
                        #the time of that last record of that session with
                        #this session
                        if cur_ats_rec.time_to_second() - self._ordered_records[(cur_ip, ip_session_tracker[cur_ip])][-1].time_to_second() > self.DEAD_SESSION_PAUSE:
                            #the session is dead we have to start a new session
                            ip_session_tracker[cur_ip] += 1
                            new_session = True

                    if new_session:
                        self._ordered_records[(cur_ip, ip_session_tracker[cur_ip])] = [cur_ats_rec]
                    else:
                        self._ordered_records[(cur_ip, ip_session_tracker[cur_ip])].append(cur_ats_rec)

                else:
                    #unable to munch and grasp the data due to unrecognizable format
                    total_failure_munches += 1

                #reporting progress
                current_progress = (self._log_lines.tell()*100)/total_file_size
                if (current_progress != previous_progress):
                    print "%", current_progress
                    previous_progress = current_progress


            self._log_lines.close()

        self._log_file_list = []

        #for debug, it should be moved to be dumped in the logger
        print "Parsed ", len(self._ordered_records)
        if total_failure_munches > 0:
            print "Failed to parse ", total_failure_munches, " records"
        self.dict_invalid = False

    def parse_log_old(self):
        """
        Read each line of the log file and batch the
        records corresponding to each (client (ip), session)
        make a dictionary of lists each consisting of all records of that session
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
