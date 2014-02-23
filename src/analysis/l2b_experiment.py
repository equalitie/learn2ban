"""
To represent an experiment run by the Analyser, tester, etc.

AUTHORS:
- Vmon (vmon@equalit.ie) Feb 2013: Initial version
- Ben (benj.renard@gmail.com) June 2013: pca, mRMR
- Vmon July 2013: Modified PCA analysis to actually rate importance of 
                  features.
"""
from sklearn import svm, decomposition, manifold
import pylab as pl
import numpy as np
import sys
import os

from analysis.mRMR import mrmr


class L2BExperiment():
    """
    Each L2BExperiment consists of a
      - Training Set and Testing Set,
      - It might have a known Test target or not in that case it (should
    be able to TODO) cross valdiate it self return an score. It can graph
    It
      - It can graph itself.
      - It (TODO) can load and store itself into the database.
    """
    def __init__(self, training_set=None, testing_set=None, experiment_trainer=None):
        """
        Simple initialization by setting the training set and the test
        set. Either can be null, the experiment can be only training or
        only testing. The classifier can be indicated during training or
        testing.
        """
        self._training_set = training_set
        self._testing_set = testing_set
        self._experiment_trainer = experiment_trainer

        #set mRMR executable
        cur_dir = os.path.dirname(__file__)
        if sys.platform == 'darwin':
            aux = 'mrmr_osx_maci_leopard'
        else:
            aux = 'mrmr'
        self.mrmr_ex = cur_dir + '/' + aux

    def train(self, experiment_trainer=None):
        """
        It make a train2ban object in the spot, set the training set and
        train.

        INPUT:
            experiment_classifier: Can be None if it set before otherwise
            raise an exception

        """
        if (experiment_trainer):
            self._experiment_trainer = experiment_trainer

        self._experiment_trainer.set_training_set(self._training_set)

        self._experiment_trainer.train()

    def predict(self, experiment_trainer=None):
        """
        Run the predict and update the _predicted_target

        INPUT:
            experiment_classifier: Can be None if it set before otherwise
            raise an exception
        """
        if (experiment_trainer):
            self._experiment_trainer = experiment_trainer

        self._predicted_target = self._experiment_trainer.predict(testing_set._ip_feature_array)

    def cross_validate_test(self, experiment_trainer=None):
        """
        Use the classifier score function to cross-validate
        the result on the test set. user has to train the
        classifier in advance

        INPUT:
            experiment_classifier: Can be None if it set before otherwise
            raise an exception
        """
        if (experiment_trainer):
            self._experiment_trainer = experiment_trainer

        if ((self._testing_set == None) or (self._testing_set._target == None)):
            raise ValueError, "A valid Testing set with known target is required"

        return self._experiment_trainer._ban_classifier.score(self._testing_set._ip_feature_array, self._testing_set._target)

    def cross_validate_score_train(self, experiment_trainer=None, no_of_repetition=1):
        """
        Uses the native sklearn cross_validation class method to split only
        the traiing set and compute the prediction score

        INPUT:
            experiment_classifier: Can be None if it set before otherwise
            raise an exception

            no_of_repetition: no of type to repeat the validaion
        """
        if (experiment_trainer):
            self._experiment_trainer = experiment_trainer

        from sklearn import cross_validation

        scores = cross_validation.cross_val_score(self._experiment_trainer._ban_classifier, self._training_set._ip_feature_array, self._training_set._target, cv=no_of_repetition)
        return (scores.mean(), scores.std())

    def pca_transform_detail(self, nb_components=0):
        """
        Compute the PCA transformation for the self._ip_feature_array with
        nb_components PCA components.

        This is used in determining the important features

        Suppose
        PCA1 = c_11*feature_1 + c12*feature_2 + ... + c1n*feature_n
        
        For now we only look at normalised coefficients
        
        |c_11|/sqrt(sum(c1i_2^2))

        and we look at how much of the variance is explained in PCA1
        If PCA1 one is small one can go to PCA2 etc.
        to retrieve c_1i we compute pca_trans(e_i)[0]
        
        INPUT:
            nb_components: the number of PCA to be used to explain the model
                           0 means as many as number features.

        OUTPUT:
           (the transfarmation matrix in np array, importance of each component)
        """
        no_of_features = self._training_set._ip_feature_array.shape[1]
        if (nb_components==0):
            nb_components = no_of_features

        whole_space = np.append(self._training_set._ip_feature_array, self._testing_set._ip_feature_array, axis=0)
        whole_target = np.append(self._training_set._target, self._testing_set._target, axis=0)

        dim_reducer = decomposition.PCA(n_components=nb_components)
        dim_reducer.fit(whole_space)

        #retrieving the coefficients by transforming identity matrix
        #each column of pca_coeffs is describing each pca components
        #in term of features, so it is (PCA_Transform)^(Trasposed)
        pca_coeffs = dim_reducer.transform(np.identity(no_of_features))

        # Tests by Ben
        return (pca_coeffs, dim_reducer.explained_variance_ratio_)

    def plot(self, dim_reduction_strategy='PCA', kernel='linear'):
        """
        Plot the result of the fiting
        """
        whole_space = np.append(self._training_set._ip_feature_array, self._testing_set._ip_feature_array, axis=0)

        whole_target = np.append(self._training_set._target, self._testing_set._target, axis=0)
        if dim_reduction_strategy == 'PCA':
            # change randomizedPCA to PCA
            # dim_reducer = decomposition.RandomizedPCA(n_components=2)
            dim_reducer = decomposition.PCA(n_components=2)
        elif dim_reduction_strategy == 'Isomap':
            n_neighbors = 30
            dim_reducer = manifold.Isomap(n_neighbors, n_components=2)
        elif dim_reduction_strategy == 'MDS':
            dim_reducer = manifold.MDS(n_components=2, n_init=1, max_iter=100)

        dim_reducer.fit(whole_space)
        reduced_train_spc = dim_reducer.transform(self._training_set._ip_feature_array)
        reduced_test_spc = dim_reducer.transform(self._testing_set._ip_feature_array)

        reduced_whole_space = np.append(reduced_train_spc, reduced_test_spc, axis=0)

        clf = svm.SVC(kernel=str(kernel), gamma=10)
        clf.fit(reduced_train_spc, self._training_set._target)

        pl.figure(0)
        pl.clf()
        pl.scatter(reduced_whole_space[:, 0], reduced_whole_space[:, 1], c=whole_target, zorder=10, cmap=pl.cm.Paired)

        # Circle out the test data
        pl.scatter(reduced_test_spc[:, 0], reduced_test_spc[:, 1],
                   s=80, facecolors='none', zorder=10)

        pl.axis('tight')
        x_min = reduced_whole_space[:, 0].min()
        x_max = reduced_whole_space[:, 0].max()
        y_min = reduced_whole_space[:, 1].min()
        y_max = reduced_whole_space[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        pl.pcolormesh(XX, YY, Z > 0, cmap=pl.cm.Paired)
        pl.contour(XX, YY, Z, colors=['k', 'k', 'k'],
              linestyles=['--', '-', '--'],
              levels=[-.5, 0, .5])

        norm_mode = (self._training_set._normalisation_data and "individual" or "sparse")
        pl.title("Norm: " + norm_mode + ", Kernel:" + kernel + ", Dimension reduced by " + dim_reduction_strategy)

        pl.show()

    def get_mrmr(self):
        """
        Gets the mRMR associated with this experiment
        """
        whole_space = np.append(self._training_set._ip_feature_array, self._testing_set._ip_feature_array, axis=0)
        fn = ['F%d' % n for n in range(len(whole_space[0]))]
    
        whole_target = np.append(self._training_set._target, self._testing_set._target, axis=0) 

        #we need to map Good to 1 and Bad to -1 Good is 0 and Bad 1 
        #x = Good y = 1
        #x = Bad  y = -1
        #y =(Good - Bad)/ (1 -(-1)) *(x - Good) + 1
        new_slope = (1-(-1))/(self._training_set.GOOD_TARGET-self._training_set.BAD_TARGET)
        mRMR_target = [new_slope*(cur_class - self._training_set.GOOD_TARGET)+1 for cur_class in whole_target]

        mrmrout = mrmr(whole_space, fn, mRMR_target, mrmrexe=self.mrmr_ex)
        R = mrmrout['mRMR']

        # print 'Order \t Fea \t Name \t Score'
        # for i in range(len(R['Fea'])):
        #     print '%d \t %d \t %s \t %f\n' % (i, R['Fea'][i], fn[R['Fea'][i]], R['Score'][i])

        return R['Score']
