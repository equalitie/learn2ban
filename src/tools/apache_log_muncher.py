import re
import glob
from os.path import dirname
from os import getcwd
months = {
'Jan':'01',
'Feb':'02',
'Mar':'03',
'Apr':'04',
'May':'05',
'Jun':'06',
'Jul':'07',
'Aug':'08',
'Sep':'09',
'Oct':'10',
'Nov':'11',
'Dec':'12'}

parts = [
    r'(?P<host>\S+)',                   # host %h
    r'(?P<user>\S+)',                   # user %u
    r'\[(?P<day>\S{2})\/(?P<month>\S{3})\/(?P<year>\S{4}):(?P<time>.+)\s.+\]',                # time %t
    r'"(?P<request>.+)"',               # request "%r"
    r'(?P<protocol>.*)',               # protocol "%{Protocol}i"
    r'(?P<referer>.*)',               # referer "%{Referer}i"
    r'(?P<status>[0-9]+)',              # status %>s
    r'(?P<size>\S+)',                   # size %b (careful, can be '-')
    r'"(?P<agent>.*)"',                 # user agent "%{User-agent}i"
    r'(?P<hit>\S+)',               # hit "%{Hit}i"
    r'(?P<type>\S+)',               # type "%{Type}i"
    r'\S+',                             # indent %l (unused)
    r'(?P<unused1>.*)',               # unused1 "%{unused1}i"
    r'(?P<unused2>.*)',               # unused2 "%{Unused2}i"
]

pattern = re.compile(r'\s+'.join(parts)+r'\s*\Z')

UNREASANABLE_SIZE = 100*1024*1024 #100MB
NORMAL_HTTP_METHODS = ["GET", "HEAD", "POST"]
VALID_HTTP_METHODS = ["OPTIONS", "GET", "HEAD", "POST", "PUT", "DELETE", "TRACE", "CONNECT", "PROPFIND", "PROPPATCH", "MKCOL", "COPY", "MOVE", "LOCK", "UNLOCK", "VERSION-CONTROL", "REPORT", "CHECKOUT", "CHECKIN", "UNCHECKOUT", "MKWORKSPACE", "UPDATE", "LABEL", "MERGE", "BASELINE-CONTROL", "MKACTIVITY", "ORDERPATCH", "ACL"]

def parse_line(line):
    m = pattern.match(line)
    if not m:
        print "failed to parse ", line
        return None
    
    res = m.groupdict()

    if res["user"] == "-":
        res["user"] = None

    res["status"] = int(res["status"])

    res["http_ver"] = res["request"].split()[-1]
    res["method"] = res["request"].split()[0]
    res["request"] = res["request"].split()[-2]

    if res["method"] not in VALID_HTTP_METHODS:
        print "ignoring request with invalid method %s in %s"%(res["method"],line)
        return None
    elif res["method"] not in NORMAL_HTTP_METHODS:
        print "abnormal http request, somebody messing around?:", line
             
    #res["type"] = res["request"].find('.', res["request"].rfind('/')) == -1 and 'html' or res["request"].split('.')[-1]

    #the specific extension is only important in text/html otherwise
    #the general type gives us enough information
    doc_type = str(res["type"]).split('/')
    if len(doc_type) == 2:
        #check for non standard
        res["type"] = str(doc_type[1]) if str(doc_type[1]) == 'html' else str(doc_type[0])
    else:
        #the doc type is not valid perhapse because the request has not been served
        #we guess the type from the request then
        res["type"] = res["request"].find('.', res["request"].rfind('/')) == -1 and 'html' or res["request"].split('.')[-1]
        
        #group all the images
        res["type"] = res["type"] in ["jpg", "gif", "png"] and "image" or res["type"]

    #if (not res["type"] == 'html'): print res["type"], line

    if res["size"] == "-":
        res["size"] = 0
    else:
        res["size"] = int(res["size"])

    #just to observe the illogically large sizes
    if (res["size"] >= UNREASANABLE_SIZE):
        print "unreasonably large payload ", line
        return None

    if res["referer"] == "-":
        res["referer"] = None

    res["time"] = res["year"]+'-'+months[res["month"]]+'-'+res["day"] + "T" + res["time"]+"Z"

    return res;

if __name__ == "__main__":
    for files in glob.glob("*.log"):
        for l in open(files,'rb'):
	    parse_line(l)
