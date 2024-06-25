#
# JAX/Lineax-based code for tridiagonal operations
#

import jax
import jax.numpy as jnp
import lineax

# tridiagonal matrix assembly from three 1D arrays

def assemble(a,b,c):
    A = jnp.vstack((c,b,a))
    A = A.at[0,-1].set(0.0)
    A = A.at[-1,0].set(0.0)
    return A

# tridiagonal matrix disassembly into three 1D arrays

def disassemble(A):
    return A[2,:], A[1,:], A[0,:]

# tridiagonal identity matrix

def identity(J):
    return jnp.vstack((jnp.zeros(J), jnp.ones(J), jnp.zeros(J)))

# tridiagonal matrix transpose

def transpose(A):
    return jnp.vstack((jnp.roll(A[2,:],-1), A[1,:], jnp.roll(A[0,:],1)))

# tridiagonal matrix-vector multiply

def multiply(A, x):
    return A[0,:]*jnp.roll(x,-1) + A[1,:]*x + A[2,:]*jnp.roll(x,1)

# vector outer product retaining only tridiagonal elements

def outer(y, x):
    A = jnp.vstack((jnp.roll(x,-1), x, jnp.roll(x,1))) * y
    A = A.at[0,-1].set(0.0)
    A = A.at[-1,0].set(0.0)
    return A

# JAX tridiagonal solve -- unfortunately does not yet support
# forward and reverse mode differentiation

def solve_jax(A, d):
    a, b, c = disassemble(A)
    d = jnp.reshape(d, (-1,1))
    x = jax.lax.linalg.tridiagonal_solve(a,b,c, d)
    x = jnp.reshape(x, -1)
    return x

# Lineax tridiagonal solve -- does support forward and reverse
# mode differentiation

def solve(A, d):
    a, b, c = disassemble(A)
    mat = lineax.TridiagonalLinearOperator(b,a[1:],c[:-1])
    x = lineax.linear_solve(mat,d)
    return x.value
