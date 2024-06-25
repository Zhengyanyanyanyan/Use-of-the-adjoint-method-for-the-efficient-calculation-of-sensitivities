#
# JAX-based code for cubic spline approximation for x(s)
# with zero derivative end conditions
#
# s  = spline points
# x  = data values
# xp = computed derivatives x'(s) 
#

import jax.numpy as jnp
import jax_trid as trid

def assemble_matrices(s, x):
    a = jnp.zeros_like(s)
    b = jnp.zeros_like(s)
    c = jnp.zeros_like(s)
    d = jnp.zeros_like(s)
    e = jnp.zeros_like(s)
    f = jnp.zeros_like(s)

    # zero derivative end conditions

    b = b.at[0].set(1.0)
    b = b.at[-1].set(1.0)

    # interior spline equations

    dsmi = 1 / (s[1:-1]-s[0:-2])
    dspi = 1 / (s[2:  ]-s[1:-1])

    a = a.at[1:-1].set(    dsmi   )
    b = b.at[1:-1].set( 2*(dsmi + dspi) )
    c = c.at[1:-1].set(    dspi   )
    d = d.at[1:-1].set(-3*dsmi**2 )
    e = e.at[1:-1].set( 3*(dsmi**2-dspi**2) )
    f = f.at[1:-1].set( 3*dspi**2 )
    
    # form tridiagonal matrices

    A = trid.assemble(a,b,c)
    B = trid.assemble(d,e,f)

    return A, B

# compute spline derivatives at spline points

def derivs(s, x):
    A, B = assemble_matrices(s, x)
    xp = trid.solve( A, trid.multiply(B, x) )
    return xp

# evaluate spline at ss

def eval(s, x, xp, ss):
    i = jnp.sum(jnp.where(s<ss,1,0));

    if (i==0):         # outside entire interval
        xx = x[0]

    elif (i==s.size):  # outside entire interval
        xx = x[-1]

    else:              # interval (s[i-1], s[i])

        ds = s[i]-s[i-1]
        t  = (ss-s[i-1]) / ds
        xx = (1   - 3*t**2 + 2*t**3) * x[i-1]     \
           + (      3*t**2 - 2*t**3) * x[i]   \
           + (  t - 2*t**2 +   t**3) * ds*xp[i-1] \
           + (    -   t**2 +   t**3) * ds*xp[i]

    return xx

