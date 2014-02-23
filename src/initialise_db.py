"""
Construct the db

AUTHORS: Bill: Initial version,
         Ben: June 2013: Storing the results, PCA, mRMR.
         Vmon: July 2013: Making tables for the regex and corresponding logs
               August 2013: Add config profiles to config table

- July 2013: Experiments should only store the log_ids for
training and testing. the regex_assignments will tell which regexes are detecting the
bots in that log. This is  mainly because one might need more regex to detect all bots in
a file. This problem was previously address by putting more than one regex in a filter.
However, applying regexes that finds bots in some attacks not necessarily will single out bots in another log

             I see no reason for not storing the regex itself in the database.
"""
from tools.learn2bantools import Learn2BanTools

l2btools = Learn2BanTools()

l2btools.connect_to_db()

l2btools.cur.execute("create table IF NOT EXISTS config (id INT NOT NULL AUTO_INCREMENT,profile_name VARCHAR(255), absolute_paths BOOLEAN , training_directory VARCHAR     (255), testing_directory VARCHAR(255), analyser_results_directory VARCHAR(255),    regex_filter_directory VARCHAR(255),default_filter_file VARCHAR(255), PRIMARY      KEY(id) ) ENGINE=INNODB;")

l2btools.cur.execute("create table IF NOT EXISTS regex_filters ( id INT NOT NULL AUTO_INCREMENT, name VARCHAR(255),    regex VARCHAR(4096), PRIMARY KEY(id)) ENGINE = INNODB;")

l2btools.cur.execute("create table IF NOT EXISTS experiments (id INT NOT NULL AUTO_INCREMENT, regex_filter_id INT,     kernel_type VARCHAR(255), training_log VARCHAR(255), testing_log VARCHAR(255), enabled BOOLEAN, comment LONGTEXT,  FOREIGN KEY(regex_filter_id) REFERENCES regex_filters(id), norm_mode VARCHAR(100), PRIMARY KEY(id) ) ENGINE = INNODB")

# created by Ben for storing additional results. In the end experiment_result should be dropped
# l2btools.cur.execute("drop table experiment_results")
l2btools.cur.execute("create table IF NOT EXISTS experiment_results( id INT NOT NULL AUTO_INCREMENT, experiment_id INT, FOREIGN KEY(experiment_id) references experiments(id), result_file VARCHAR(255), proportion FLOAT, score FLOAT, active_features VARCHAR(255), pca_ratios VARCHAR(255), mrmr_score VARCHAR(255), PRIMARY KEY(id) )")

#Keeping track of the name of training logs in the db
l2btools.cur.execute("create table IF NOT EXISTS logs( id INT NOT NULL AUTO_INCREMENT, file_name VARCHAR(255), note LONGTEXT, PRIMARY KEY(id) )")

l2btools.cur.execute("create table IF NOT EXISTS regex_assignment( id INT NOT NULL AUTO_INCREMENT, regex_filter_id INT, log_id INT, FOREIGN KEY (log_id) REFERENCES logs(id), FOREIGN KEY(regex_filter_id) REFERENCES regex_filters(id), PRIMARY KEY(id) )")

l2btools.cur.execute("create table IF NOT EXISTS experiment_logs( id INT NOT NULL AUTO_INCREMENT, experiment_id INT, log_id INT, FOREIGN KEY (experiment_id) REFERENCES experiments(id), FOREIGN KEY (log_id) REFERENCES logs(id), PRIMARY KEY(id) )")

l2btools.cur.execute("Insert into config( training_directory, testing_directory, analyser_results_directory,regex_filter_directory,default_filter_file) values ('/data/training/','/data/testing/','/analysis/results_dir/', '/data/filters/', 'regex_filters.xml')")

#l2btools.cur.execute("Insert into regex_filters( name, filter_file) values ('User Agent', 'regex_filters.xml')")
#l2btools.db.commit()

l2btools.cur.execute("Insert into experiments(regex_filter_id,kernel_type, training_log, testing_log,    norm_mode) values (1,'linear', 'training.log', 'testing.log','sparse')")
l2btools.cur.execute("Insert into experiments(regex_filter_id,kernel_type, training_log, testing_log,    norm_mode) values (1,'linear', 'training.log', 'testing.log','individual')")
l2btools.cur.execute("Insert into experiments(regex_filter_id,kernel_type, training_log, testing_log,    norm_mode) values (1,'linear', 'training.log', 'testing.log','sparse')")
l2btools.cur.execute("Insert into experiments(regex_filter_id,kernel_type, training_log, testing_log,    norm_mode) values (1,'linear', 'training.log', 'testing.log','individual')")

l2btools.cur.execute("Insert into experiments(regex_filter_id,kernel_type, training_log, testing_log,    norm_mode) values (1,'linear', 'training.log', 'testing.log','sparse')")
l2btools.cur.execute("Insert into experiments(regex_filter_id,kernel_type, training_log, testing_log,    norm_mode) values (1,'linear', 'training.log', 'testing.log','individual')")

l2btools.db.commit()

l2btools.disconnect_from_db()
