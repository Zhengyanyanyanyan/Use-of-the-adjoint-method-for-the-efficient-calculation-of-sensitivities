#
# JAX-based test code to validate tridiagonal sensitivities
#

import jax
import jax.numpy  as jnp
import jax.random as random
from jax import jvp, vjp

import jax_trid   as trid   # my tridiagonal module

jax.config.update("jax_enable_x64", True)  # force 64-bit accuracy

# dot product function for vectors and arrays

def dp(A_b,A_d):
    return jnp.sum(jnp.multiply(A_b,A_d))

# tridiagonal test matrix A and random r.h.s. b

N = 3;
ones = jnp.ones(N)
A = jnp.vstack( (-1.0*ones, 2.0*ones, -1.0*ones) )
A = A.at[0,-1].set(0.0)
A = A.at[-1,0].set(0.0)

key = random.key(123456789)
key, sub = random.split(key); b = random.normal(sub,(N,))

x  = trid.multiply(A, b)
b2 = trid.solve(A, x)

print('tridiagonal solve/multiply check:')
print('error: %g \n' % jnp.linalg.norm(b-b2))

# random matrices for forward/reverse sensitivity checks

key, sub = random.split(key); A_d = random.normal(sub,(3,N))
A_d = A_d.at[0,-1].set(0.0)
A_d = A_d.at[-1,0].set(0.0)

key, sub = random.split(key); b_d = random.normal(sub,(N,))
key, sub = random.split(key); x_b = random.normal(sub,(N,))

# forward/reverse multiplication checks

#x   = trid.multiply(A, b)
#x_d = trid.multiply(A, b_d) + trid.multiply(A_d, b)
#b_b = trid.multiply(trid.transpose(A), x_b)
#A_b = trid.outer(x_b,b)

x, x_d = jvp(trid.multiply, (A,b), (A_d,b_d))

x, f_vjp = vjp(trid.multiply, A,b)
A_b, b_b = f_vjp(x_b)

print('tridiagonal multiply, forward/reverse sensitivity check')
print('error: %g \n' % (dp(x_b,x_d) - dp(A_b,A_d) - dp(b_b,b_d)))

# forward/reverse solution checks

#x   = trid.solve(A, b)
#x_d = trid.solve(A, b_d-trid.multiply(A_d, x))
#b_b = trid.solve(trid.transpose(A), x_b)
#A_b = - trid.outer(b_b,x)

x, x_d = jvp(trid.solve, (A,b), (A_d,b_d))

x, f_vjp = vjp(trid.solve, A,b)
A_b, b_b = f_vjp(x_b)

print('tridiagonal solve, forward/reverse sensitivity check')
print('error: %g \n' % (dp(x_b,x_d) - dp(A_b,A_d) - dp(b_b,b_d)))
