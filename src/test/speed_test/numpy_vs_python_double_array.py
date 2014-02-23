"""
This is to compare how fast numpy or python filling up their double array

AUTHORS:
- Vmon (vmon@equalit.ie) 2012: initial version.

"""

import numpy as np
from time import clock

test_size = 3*10**3
print "Testing an array of size %ix%i"%(test_size,test_size)
#initialization
t0 = clock()
py_double = [[x for x in xrange(0,test_size)] for y in xrange(0,test_size)]
t1 = clock()
print "py init:",(t1-t0)

t0 = clock()
numpy_double = np.zeros((test_size, test_size))
t1 = clock()
print "numpy init:",(t1-t0)

t0 = clock()
for i in xrange(0,test_size):
    for j in xrange(0,test_size):
        py_double[i][j] = i * test_size + j
t1 = clock()
print "py fillup:",(t1-t0)

t0 = clock()
for i in xrange(0,test_size):
    for j in xrange(0,test_size):
        numpy_double[i][j] = i * test_size + j
t1 = clock()
print "numpy fillup:",(t1-t0)
