Learn2ban
=========
Open source machine learning DDOS detection tool

Copyright 2013 eQualit.ie

Learn2ban is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see `<http://www.gnu.org/licenses/>`.

Installation
============

The following libraries should be installed

    [sudo] apt-get install libmysqlclient-dev
    [sudo] apt-get install build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev
    [sudo] apt-get install python-matplotlib
    easy_install pip

Install required packages

    pip install -r requirements.txt

Initialize Learn2ban training database

    python src/initialise_db.py
    
Testing:
--------
Run python unit tests in the Learn2ban/src/test/ directory to ensure functionality.

Configuration
=============

Access to mysql server
----------------------
User needs to enter the access detail for a mysql server in config/train2ban.cfg. For example:

    db_user = root
    db_password = thisisapassword
    db_host = mydb.myserver.com
    db_name = learn2ban
    config_profile = myconfig

Then user need to run initialise_db.py

    python initialise_db.py

To create the database. User then is required to make a record in config table with profile name equal to config_profile (myconfig in this example) and enter the relevant directories in the table.

Regex filters
-------------

In order to annotate input logs, Learn2ban uses the fail2ban regex filtering system to mark IP addresses as malicious or legitimate. The regex rules to apply can be added to regex_filter table in learn2ban database

Training data
-------------
The data from which the Learn2ban SVM model will be constructed should be placed in the directory defined in the profile or entered in absolute path in experiment table if the profile asks for absolute path.

Running Learn2ban Experiments
=============================

Learn2ban is currently designed to run in an experimental mode to allow users to create multiple variations of models, based on their training data, and to easily analyze the efficacy and accuracy of these models.

User needs to enter the log file names in logs table, assign the regexes which identifies the bots in the log in regex_assignment table. User then design an experiment in experiments table, and assign the log to it in experiment_logs. 

To run a configured learn2ban experiment, enable the experiment in experiments table and execute

    python src/analysis/experimentor.py

Learn2ban model feature set
===========================
In order to classify requesting IP addresses as legitimate or malicious the Learn2ban SVM model takes into account the following set of features derived from HTTP log data.

These features are implemented at Learn2ban/src/features.

* average_request_interval - this feature considers the behaviour of the requester in terms of the average number of request made within a given interval. This is essentially the frequency with which a requester attempts to access a given host. It takes into account the requests as whole not merely in terms of a single page.
* cycling_user_agent - a common attack for DDOS Botnets is to change user agent repeatedly during an attack. This strategy can be quite effective against even the most generalised regex rules. If the IP never repeats its user agent then rules put in place to block requesters using obscure user agents will still be subverted. In the context of a real human user, or even a spider bot, user agent rotation is highly aberrant.
* html_to_image_ratio - This feature considers the type of content that is being requested. It considers if a requester is only retrieving HTML content but no ancillary data such as images, css or javascript files.
* variance_request_interval - While many DDOS attacks use a very simplistic brute force approach, some have incorporated a slightly more sophisticated approach by making burst requests in order to avoid being blocked by simple rules which allow only a certain number of requests within a time frame.
* payload_size_average - this feature looks at the size of the content that a requester is retrieving.
* HTTP_response_code_rate - Considers http response rate, primarily looking for error codes that may signal a cache busting attack.
* request_depth - Normal site users with commonly browse beyond the home page of a given site. Human users interaction with a website will resemble browsing more than that of a botnet.
* request_depth_std - As an adjunct to request depth, this feature considers the standard deviation of a bot's request.
* session_length - This feature also elucidates general behaviour considering the requester's interaction with a given sight in terms of session time.
* percentage_consecutive_requests - To further elucidate the requester's interaction with a given site we additionally consider how many of the requests made were consecutive as another window onto frequency.

Adding new features
===================

It is possible to easily extend Learn2ban's feature set by inheriting from the prototype feature at Lear2ban/src/features/learn2ban_feature.py.

The new feature needs to register the log data index of the feature under consideration and implement the compute() method which will return the feature value.

This project forms part of the [Deflect](https://deflect.ca) project.
