"""
This class is used to hold train as well as test set

AUTHORS:

  - vmon (vmon@equaliti.e) 2013: Moved from train2ban

"""
import numpy as np
from operator import itemgetter

import pdb
class TrainingSet:
    """
    Each TrainingSet consists of data, target and ip_index, in particular
    you shouldn't add ip to the data without also adding it into the index
    so it made sense to make class to ploice that
    """
    BAD_TARGET = 1
    GOOD_TARGET = 0
    MAX_FEATURE_INEDX = 10 #TODO: This feels embaressingly static but just for now
    #improving readiblity of normalisation
    NORMALISATION_TYPE = 0
    SAMPLE_MEAN = 1
    SAMPLE_STD = 2

    def __init__(self):
        """
        Intitilizes the empty lists
        """
        self._ip_feature_array = np.array([])
        self._ip_index = [] #this keep trac of which sample row belong to
                            #which IP

        self._sample_index = {} #this gets an ip and return its place in
                                #the sample list, of course I can use find on
                                #_ip_index but that's no longer O(1). on the
                                #other hand I can decide bad from good during
                                #add feature process but don't want to limit
                                #the user.
        self._target = np.array([])

        self._normalisation_function = None
        self._normalisation_data = None

    @classmethod
    def _construct_training_set(cls, ip_feature_array, target, ip_index, sample_index, normalisation_function, normalisation_data):
        """
        Semi-private constructor, not meant for public usage which
        intitilizes the based on lists that are already initiated and
        consistant
        """
        self_inst = cls()
        self_inst._ip_feature_array = ip_feature_array
        self_inst._target = target

        #indexes
        self_inst._ip_index = ip_index
        self_inst._sample_index = sample_index

        #normalisation
        self_inst._normalisation_function = normalisation_function
        self_inst._normalisation_data = normalisation_data

        return self_inst

    def __len__(self):
        """
        Defines what is commonly understood of size of a training set
        """
        #sanity check
        assert(self._ip_feature_array.shape[0] == len(self._ip_index))
        return self._ip_feature_array.shape[0]

    def normalise_sparse(self, test_array=None):
        """
        Normalises based by making each record to have unit vector feature

        INPUT:
            test_array: the test array to be normalised, if None, then
                        normalise the self._ip_feature_array
        """
        #stupid that numpy doesn't have something symmetrical to colunmn-wise
        #norm division or I don't it
        if (test_array == None):
            array_2b_normal = self._ip_feature_array
            self._normalisation_function = self.normalise_sparse
        else:
            array_2b_normal = test_array

        #if the array is empty then there is nothing to normalise
        if array_2b_normal.size == 0:
            return array_2b_normal

        array_2b_normal = array_2b_normal / (np.repeat(np.apply_along_axis(np.linalg.norm, 1 , array_2b_normal), array_2b_normal.shape[1])).reshape(array_2b_normal.shape)

        self.normalisation_data = ['sparse'] #no more info needed

        #dealing with python retardation
        if (test_array == None):
            self._ip_feature_array = array_2b_normal
        else:
            return array_2b_normal

    def normalise_individual(self, test_array = None, redundant_feature_reduction = True):
        """
        Normalises based on standardising the sample

        INPUT:            test_array: the test array to be normalised, if None, then
                        normalise the self._ip_feature_array
        """
        #If we don't have normalisation data on file we have
        #to generate it and store it
        if (test_array == None):
            self._normalisation_function = self.normalise_individual
            array_2b_normal = self._ip_feature_array

            #We need to remember these data to use during prediction
            #to keep uniformity between normalisation strategy we
            #store std and mean in a list
            self._normalisation_data = [ \
                'individual', \
                self._ip_feature_array.mean(axis=0), \
                self._ip_feature_array.std(axis=0)]
        else:
            array_2b_normal = test_array


        #if the array is empty then there is nothing to normalise
        if array_2b_normal.size == 0:
            return array_2b_normal

        #DON'T DO THAT CARELESSLY
        #because during the prediction you need to kick them out and you might
        #not have the info to do that. It is OK for testing but not for all the time

        #we kick out features which are the same for every
        #entery hence has no effect on the training
        if (redundant_feature_reduction):
            dimension_reducer = [cur_feature_std != 0  for cur_feature_std in self._normalisation_data[self.SAMPLE_STD]]
        else:
            #dimension_reducer become only a copier
            dimension_reducer = [True for cur_feature_std in self._normalisation_data[self.SAMPLE_STD]]

        reduced_std = self._normalisation_data[self.SAMPLE_STD][np.where(dimension_reducer)]
        reduced_mean = self._normalisation_data[self.SAMPLE_MEAN][np.where(dimension_reducer)]

        array_2b_normal = array_2b_normal[:,[red_plc[0] for red_plc in enumerate(dimension_reducer) if red_plc[1]]]

        array_2b_normal = (array_2b_normal - reduced_mean)/reduced_std

        #dealing with python retardation
        if (test_array == None):
            self._ip_feature_array = array_2b_normal
        else:
            return array_2b_normal

    @classmethod
    def fromPickle(cls, filename):
	classifier = joblib.load(filename)
	return cls(classifier)

    def add_ip(self, new_ip_session, ip_features):
        """
        Insert the ip in the index as well as in the data set and
        the ip dict.

        IP repeatition raises error.

        (why? on one hand it could be that an IP appears in two logs, on the
        other hand its behavoir might be different in two logs but in general we
        do not consider that cause it might be that an IP is used as bot and the
        user but we mark the IP either good or bad. The best way is two have
        way of updating the features with new values. That is necessary when
        we are live and we don't want to compute all values from the begining.
        but for now I just ignore the second repetition

        INPUT:
            new_ip: string rep of the ip to be added
            ip_features: a list of features corresponding to new_ip
        """
        if new_ip_session in self._sample_index:
            raise ValueError, "The IP dict has an IP stored already in the trainer set"

        #This way we let some feature not to be computed and not to be part of
        #model fiting. (TODO is it worth it)

        #having this loop only make sense when we are cherry picking features
        #if we are using the whole feature set then you can just copy the list or
        #something. So, I'm doing that till we implement the cherry picking
        #mechanism
        if (len(self._ip_feature_array) == 0):
            self._ip_feature_array = np.array([[0]*(self.MAX_FEATURE_INEDX)])

        else:
            self._ip_feature_array = np.append(self._ip_feature_array,[[0]*(self.MAX_FEATURE_INEDX)],axis=0)

        for cur_feature in ip_features:
            self._ip_feature_array[-1][cur_feature-1] = ip_features[cur_feature]

        #something like this is more desirable for the sake of speed
        #but that need changing the feature gathering TODO?
        #np.append(self._ip_feature_array,ip_features[1:], axis=0)

        #turned out doesn't work because features are gathered in dic of
        #dics so the first compromise is

        # self._ip_feature_array = len(self._ip_feature_array) and \
        #     np.append(self._ip_feature_array,[map(itemgetter(1),sorted(ip_features.items(), key = itemgetter(0)))], axis=0) or \
        #     np.array([map(itemgetter(1),sorted(ip_features.items(), key = itemgetter(0)))])

        #this approach doesn't work because some features can't be computed
        #and then we get dim error which is legit

        #make it two dimensional
        #if (self._ip_feature_array.ndim == 1): self._ip_feature_array = self._ip_feature_array.reshape(1,self._ip_feature_array.shape[0])

        self._sample_index[new_ip_session] = len(self._ip_index)
        self._ip_index.append(new_ip_session)

    def dump_data(self):
        """
        Just empties all the training set and starts as new
        """
        self._ip_feature_array =  np.array([])
        self._ip_index = []
        self._sample_index = []
        self._target = np.array([])

        #forget about normalisation
        self._normalisation_function = None
        self._normalisation_data = None

    def initiate_target(self):
        """
        Indicates that we are done with adding ips and we are ready to set
        targets. This will setup a target as big as len(_ip_feature_list)

        if the target is already initialized it expands so it match the
        desired length
        """
        self._target = np.repeat(self.GOOD_TARGET,self._ip_feature_array.shape[0])

    def mark_as_bad(self, bad_ip):
        """
        Searches for the ip and set its target as bad. _target has to be
        initialized in advance
        """
        cur_session = 0
        while((bad_ip, cur_session) in self._sample_index) :
            self._target[self._sample_index[(bad_ip, cur_session)]] = self.BAD_TARGET
            cur_session += 1

    def no_culprit(self):
        """
        return True if there is no bad target is set, such a set
        shouldn't go to training
        """
        return sum(self._target) == self.GOOD_TARGET * len(self._target)

    def precook_to_predict(self, ip_feature_db):
        ip_set = TrainingSet._construct_training_set( \
            ip_feature_array = np.array([]), \
                target = np.array([]), \
                ip_index = [], \
                sample_index = {}, \
                normalisation_function = self._normalisation_function, \
                normalisation_data = self._normalisation_data)

        for cur_ip in ip_feature_db:
            ip_set.add_ip(cur_ip, ip_feature_db[cur_ip])

        ip_set._ip_feature_array =  ip_set._normalisation_function(test_array = ip_set._ip_feature_array)

        return ip_set

    def get_training_subset(self, case_selector = None, feature_selector=None):
        """
        This helps to gets a subset of data to be used for training, inform of
        net training set.

WARNING: feature_selector and cutting feature DOES NOT work for now
        because it messes up normalisation!!!! (TODO)

        INPUT:
            case_selector: a list of boolean element the size of
                           number of rows in the training set (otherwise
                           raises an exception) True means consider the
                           record in training. None means everything

            feature_selector: a list of boolean element the size of
                           number of columns in the training set (otherwise
                           raises an exception) True means consider the
                           feature in training. None means everything

        """
        #if _ip_feature_array is empty there is nothing to do
        #we may as way raise an exception
        if self._ip_feature_array.size == 0:
            raise ValueError, "Not able to subset an empty set"

        if (case_selector == None):
            case_selector = np.repeat(True, self._ip_feature_array.shape[0])

        if (feature_selector == None):
            feature_selector = np.repeat(True, self._ip_feature_array.shape[1])

        if ((len(case_selector) != self._ip_feature_array.shape[0]) or \
            (len(feature_selector) != self._ip_feature_array.shape[1])):
            raise ValueError, "The dimension of subset selector does not match the dimension of the trainig set"

        map(bool, list(case_selector)) #getting of rid ambiguity
        map(bool, list(feature_selector))

        subset_selector = (np.repeat(feature_selector, len(case_selector))*np.repeat(case_selector, len(feature_selector))).reshape(self._ip_feature_array.shape)

        training_feature_subset = self._ip_feature_array[np.where(subset_selector)].reshape(sum(list(case_selector)),sum(list(feature_selector)))
        subtarget = self._target[np.where(case_selector)]

        #The only problem is that the indexing to ips isn't valid anymore
        #We need to build an index translation table
        i = 0 #fullset index
        j = 0 #subset index
        subset_sample_index = {}
        while(i < len(self._ip_index)):
            if (case_selector[i]):
                subset_sample_index[self._ip_index[i]]=j
                j+=1

            i+=1

        #This might be severely inefficient
        subset_ip_index = list(np.array(self._ip_index)[np.where(case_selector)])

        return TrainingSet._construct_training_set(training_feature_subset, \
                                                       subtarget, \
                                                       subset_ip_index, \
                                                       subset_sample_index, \
                                                       self._normalisation_function, \
                                                       self._normalisation_data)
