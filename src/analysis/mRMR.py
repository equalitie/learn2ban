'''
This is a python wrapper around Peng's mRMR algorithm.

mRMR is the min redundancy max relevance feature selection algorithm by
Hanchuan Peng. See http://penglab.janelia.org/proj/mRMR for more details about
the code and its author, as well as the sources and the license.

Author: Brice Rebsamen
Version: 0.1
Released on: June 1st, 2011


Copyright 2011 Brice Rebsamen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import numpy as np
import os
from subprocess import Popen, PIPE
import tempfile


def _savemrmrdatafile(data, featNames, classNames):
    '''
   Save the data to a CSV file in the format required by mRMR.
   - first row is the name of the features.
   - first col is the class names.
   - data is organized, with a sample per row.
   Returns the filename (a temporary file with the .csv extension).
   '''

    f = tempfile.NamedTemporaryFile(suffix='.csv', prefix='tmp_mrmr_', delete=False, mode='w')
    f.write(','.join(['class']+featNames)+os.linesep)
    data = np.asarray(data)
    for i in range(data.shape[0]):
        f.write(','.join([str(classNames[i])]+[str(d) for d in data[i, :]])+os.linesep)
    f.close()
    return f.name


def mrmr(data, featNames, classNames, threshold=None, nFeats=None, selectionMethod='MID', mrmrexe='./mrmr'):
    '''
   A wrapper around the mrmr executable.

   Arguments:
       data: a 2D array (size NxF)
       featNames: list of feature names (F elements)
       classNames: list of class names (N elements)

   Optional Arguments:
       threshold: data must be discrete or discretized. The default value (None)
               assumes that the data has already been discretized. Otherwise
               it has to be discretized as below u-t*s, above u+t*s or between:
               -1, +1 or 0, where u is the mean, s the standard deviation and
               t the threshold. This is done feature by feature.
       nFeats: the number of feature to select. If not given it defaults to all
               features. This will only sort the features.
       selectionMethod: either 'MID' or 'MIQ'. Default is 'MID'
       mrmrexe: the path to the mrmr executable. Defaults to './mrmr'

   Returns:
       A dictionnary with 2 elements: MaxRel and mRMR, which are the 2 results
       returned by mrmr (the 2 different feature selection criterions).
       Each is a dictionnary, with fields Fea and Score, holding the feature
       numbers and the scores respectively.

   Example:
       Generate some data: 200 samples, 2 classes, 7 features, the 2 first
       features are correlated with the class label, the 5 others are
       irrelevant. Feature names (fn) with a capital F are the relevant
       features.
       >>> N = 100
       >>> data = np.r_[ np.random.randn(N,2)+2, np.random.randn(N,2)-2 ]
       >>> data = np.c_[ data, np.random.randn(N*2,5) ]
       >>> c = [1]*N+[-1]*N
       >>> fn = ['F%d' % n for n in range(2)] + ['f%d' % n for n in range(5)]

       Pass to the mRMR program
       >>> mrmrout = mrmr(data, fn, c, threshold=0.5)

       Get the result:
       >>> R = mrmrout['mRMR']
       >>> print 'Order \t Fea \t Name \t Score'
       >>> for i in range(len(R['Fea'])):
       ...     print '%d \t %d \t %s \t %f\n' % \
       ...           (i, R['Fea'][i], fn[R['Fea'][i]], R['Score'][i])
       ...

       Order    Fea     Name    Score
       0        1       F1      0.131000
       1        0       F0      0.128000
       2        4       f4      -0.008000
       3        0       f0      -0.009000
       4        3       f3      -0.010000
       5        1       f1      -0.013000
       6        2       f2      -0.015000
   '''

    data = np.asarray(data)
    N, M = data.shape

    if nFeats is None:
        nFeats = M
    else:
        assert nFeats <= M

    mrmrexe = os.path.abspath(mrmrexe)

    assert os.path.exists(mrmrexe) and os.access(mrmrexe, os.X_OK)

    # Save data to a temporary file that can be understood by the mrmr binary
    fn = _savemrmrdatafile(data, featNames, classNames)

    # Generate the command line. See the help of mrmr for info on options
    cmdstr = mrmrexe
    cmdstr += ' -i %s -n %d -s %d -v %d -m %s' % (fn, nFeats, N, M, selectionMethod)
    if threshold is not None:
        assert threshold > 0
        cmdstr += ' -t ' + str(threshold)

    # Call mrmr. The result is printed to stdout.
    mrmrout = Popen(cmdstr, stdout=PIPE, shell=True).stdout.read().split('\n')

    # delete the temporary file
    os.remove(fn)

    # A function to parse the result
    def extractRes(key):
        Fea = []
        Score = []
        state = 0
        for l in mrmrout:
            if state == 0:
                if l.find(key) != -1:
                    state = 1
            elif state == 1:
                state = 2
            elif state == 2:
                if l == '':
                    break
                else:
                    n, f, fn, s = l.split(' \t ')
                    Fea.append(int(f)-1)
                    Score.append(float(s))
        return {'Fea': np.asarray(Fea), 'Score': np.asarray(Score)}

    # Return a dictionnary holding the features and their score for both the
    # MaxRel and mRMR criterions
    return {'MaxRel': extractRes('MaxRel features'),
            'mRMR': extractRes('mRMR features')}


if __name__ == '__main__':
    # Make some data
    N = 100
    data = np.c_[np.r_[np.random.randn(N, 5)+2, np.random.randn(N, 5)-2], np.random.randn(N*2, 30)]
    c = [1]*N+[-1]*N
    fn = ['F%d' % n for n in range(5)] + ['f%d' % n for n in range(30)]
    assert data.shape == (len(c), len(fn))

    mrmrout = mrmr(data, fn, c, threshold=0.5)

    R = mrmrout['mRMR']
    print 'Order \t Fea \t Name \t Score'
    for i in range(len(R['Fea'])):
        print '%d \t %d \t %s \t %f' % \
              (i, R['Fea'][i], fn[R['Fea'][i]], R['Score'][i])
