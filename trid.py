#
# this file defines several tridiagonal operations based on
# matrices specified in the scipy.linalg.solve_banded format;
# see https://docs.scipy.org/doc/scipy/reference/linalg.html
#
# in particular, it defines forward and reverse mode sensitivity
# calculations, as discussed in Sections 3.1.1 and 3.1.3
#

import numpy as np
import scipy

# tridiagonal matrix assembly from three 1D arrays

def assemble(a,b,c):
    A = np.vstack((np.roll(c,1), b, np.roll(a,-1)))
    A[0,0] = 0;  A[-1,-1] = 0
    return A

# tridiagonal matrix disassembly into three 1D arrays

def disassemble(A):
    a = np.roll(A[2,:],1)
    b =         A[1,:]
    c = np.roll(A[0,:],-1)
    return a, b, c

# tridiagonal identity matrix

def identity(J):
    return np.vstack((np.zeros(J), np.ones(J), np.zeros(J)))

# tridiagonal matrix transpose

def transpose(A):
    At = np.vstack((np.roll(A[2,:],1), A[1,:], np.roll(A[0,:],-1)))
    At[0,0] = 0;  At[-1,-1] = 0
    return At

# tridiagonal matrix-vector multiply

def multiply(A, x):
    A[0,0] = 0;  A[-1,-1] = 0
    b = np.roll(A[0,:]*x,-1) + A[1,:]*x + np.roll(A[2,:]*x,1)
    return b

# vector outer product retaining only tridiagonal elements

def outer(y, x):
    A = np.vstack((np.roll(y,1), y, np.roll(y,-1))) * x
    A[0,0] = 0;  A[-1,-1] = 0
    return A

# tridiagonal solve

def solve(A, b):
    x = scipy.linalg.solve_banded((1,1), A, b)
    return x

# tridiagonal solve with forward mode sensitivity

def solve_forwards(A, A_d, b, b_d):
    x   = scipy.linalg.solve_banded((1,1), A, b)
    x_d = scipy.linalg.solve_banded((1,1), A, b_d-multiply(A_d,x))
    return x, x_d

# tridiagonal solve with reverse mode sensitivity

def solve_reverse(A, b, x_b):
    x   = scipy.linalg.solve_banded((1,1), A, b)
    b_b = scipy.linalg.solve_banded((1,1), transpose(A), x_b)
    A_b = - outer(b_b,x)
    return x, A_b, b_b
