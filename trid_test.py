import numpy as np
import trid

# dot product function for vectors and arrays

def dp(A_b,A_d):
    return np.sum(np.multiply(A_b,A_d))

# tridiagonal test matrix A and random r.h.s. b

N = 10; ones = np.ones(N)
A = np.vstack( (-1.0*ones, 3.0*ones, -2.0*ones) )
A[0,0] = 0.0; A[-1,-1] = 0.0

b = np.random.rand(N)
b_copy = b.copy()

x = trid.solve(A, b)
b = trid.multiply(A, x)

print('tridiagonal solve/multiply check:')
print('error: %g \n' % np.linalg.norm(b-b_copy))

# random matrices for forward/reverse sensitivity checks

A_d = np.random.rand(3,N); A_d[0,0] = 0.0; A[-1,-1] = 0.0
b_d = np.random.rand(N)
x_b = np.random.rand(N)

# forward/reverse multiplication checks

x   = trid.multiply(A, b)
x_d = trid.multiply(A, b_d) + trid.multiply(A_d, b)

b_b = trid.multiply(trid.transpose(A), x_b)
A_b = trid.outer(x_b,b)

print('tridiagonal multiply, forward/reverse sensitivity check')
print('error: %g \n' % (dp(x_b,x_d) - dp(A_b,A_d) - dp(b_b,b_d)))

# forward/reverse solution checks

x   = trid.solve(A, b)
x_d = trid.solve(A, b_d-trid.multiply(A_d, x))

b_b = trid.solve(trid.transpose(A), x_b)
A_b = - trid.outer(b_b,x)

print('tridiagonal solve, forward/reverse sensitivity check')
print('error: %g \n' % (dp(x_b,x_d) - dp(A_b,A_d) - dp(b_b,b_d)))
