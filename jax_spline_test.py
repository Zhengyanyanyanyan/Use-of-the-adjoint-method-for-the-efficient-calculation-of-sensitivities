#
# test code to validate JAX-based spline routines
#

import jax
import jax.numpy as jnp
import jax.random as random
from jax import jvp, vjp

import jax_spline as spline

jax.config.update("jax_enable_x64", True)  # force 64-bit accuracy

# dot product function for vectors and arrays

def dp(A_b,A_d):
    return jnp.sum(jnp.multiply(A_b,A_d))

# create random test vectors

N = 6
s    = jnp.linspace(0.0,1.0,N)  # spline points
key = random.key(123456789)
key, sub = random.split(key); x    = random.normal(sub,(N,))
key, sub = random.split(key); x_d  = random.normal(sub,(N,))
key, sub = random.split(key); x_b  = random.normal(sub,(N,))
key, sub = random.split(key); xp_b = random.normal(sub,(N,))

# compute forward and reverse sensitivities for spline creation

xp, xp_d = jvp(spline.derivs, (s,x), (0.0*s,x_d))

xp, f_vjp = vjp(spline.derivs, s,x)
s_b, x_b2 = f_vjp(xp_b)
x_b2 = x_b + x_b2

print('spline creation, forward/reverse sensitivity check')
print('adj error: %g \n' % (dp(x_b,x_d) + dp(xp_b,xp_d) - dp(x_b2,x_d)))

# compute forward and reverse sensitivities for spline evaluation

key, sub = random.split(key); ss   = random.uniform(sub,(1,))  # uniform on (0,1)
key, sub = random.split(key); ss_d = random.normal(sub,(1,))
key, sub = random.split(key); xx_b = random.normal(sub,(1,))

x_b  = jnp.zeros_like(x)
xp_b = jnp.zeros_like(x)
ss_b = 0.

xx, xx_d = jvp(spline.eval, (s,x,xp,ss), (0.0*s,x_d,xp_d,ss_d))

xx, f_vjp = vjp(spline.eval, s,x,xp,ss)
s_b, x_b, xp_b, ss_b = f_vjp(xx_b)

print('spline evaluation, forward/reverse sensitivity check')
print('adj error: %g \n' % (dp(xx_b,xx_d)-dp(ss_b,ss_d)-dp(x_b,x_d)-dp(xp_b,xp_d)))
