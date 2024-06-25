# construct cubic spline approximation for x(s)
# with zero derivative end conditions
#
# s  = spline points
# x  = data values
# xp = computed derivatives x'(s) 
#

import numpy as np
import trid

def assemble_matrices(s, x):
    a = np.zeros_like(s)
    b = np.zeros_like(s)
    c = np.zeros_like(s)
    d = np.zeros_like(s)
    e = np.zeros_like(s)
    f = np.zeros_like(s)

    # zero derivative end conditions

    b[0] = 1;  b[-1] = 1;

    # interior spline equations

    dsmi = 1 / (s[1:-1]-s[0:-2])
    dspi = 1 / (s[2:  ]-s[1:-1])

    a[1:-1] =    dsmi
    b[1:-1] = 2*(dsmi + dspi)
    c[1:-1] =    dspi
    d[1:-1] = -3*dsmi**2
    e[1:-1] = 3*(dsmi**2-dspi**2)
    f[1:-1] =  3*dspi**2
    
    # form tridiagonal matrices

    A = trid.assemble(a,b,c)
    B = trid.assemble(d,e,f)

    return A, B

# compute spline derivatives at spline points

def derivs(s, x):
    A, B = assemble_matrices(s, x)
    xp = trid.solve( A, trid.multiply(B, x) )
    return xp

# compute reverse mode sensitivities at spline points

def derivs_reverse(s, x, x_b, xp_b):
    A, B = assemble_matrices(s, x)
    x_b2 = x_b + trid.multiply(trid.transpose(B), \
                    trid.solve(trid.transpose(A), xp_b) )
    return x_b2

# evaluate spline at ss

def eval(s, x, xp, ss):
    i = np.sum(np.where(s<ss,1,0));

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

# evaluate spline and forward mode sensitivity

def eval_forward(s, x, x_d, xp, xp_d, ss, ss_d):
    i = np.sum(np.where(s<ss,1,0));

    if (i==0):         # outside entire interval
        xx_d = x_d[0]

    elif (i==s.size):  # outside entire interval
        xx_d = x_d[-1]

    else:              # interval (s[i-1], s[i])
        ds   = s[i]-s[i-1]
        t    = (ss-s[i-1]) / ds
        t_d  =  ss_d / ds;
        xx_d = (1   - 3*t**2 + 2*t**3) * x_d[i-1]     \
             + (      3*t**2 - 2*t**3) * x_d[i]       \
             + (  t - 2*t**2 +   t**3) * ds*xp_d[i-1] \
             + (    -   t**2 +   t**3) * ds*xp_d[i]   \
             + ( (  - 6*t + 6*t**2) * x[i-1]          \
               + (    6*t - 6*t**2) * x[i]            \
               + (1 - 4*t + 3*t**2) * ds*xp[i-1]      \
               + (  - 2*t + 3*t**2) * ds*xp[i] ) * t_d

    return xx_d

# evaluate spline and reverse mode sensitivity

def eval_reverse(s, x, x_b, xp, xp_b, ss, ss_b, xx_b):
    i = np.sum(np.where(s<ss,1,0));

    if (i==0):         # outside entire interval
        x_b[0] = x_b[0] + xx_b

    elif (i==s.size):  # outside entire interval
         x_b[-1] = x_b[-1] + xx_b

    else:              # interval (s[i-1], s[i])
        ds   = s[i]-s[i-1]
        t    = (ss-s[i-1]) / ds
        x_b[i-1]  = x_b[i-1]  + (1   - 3*t**2 + 2*t**3) * xx_b
        x_b[i]    = x_b[i]    + (      3*t**2 - 2*t**3) * xx_b
        xp_b[i-1] = xp_b[i-1] + (  t - 2*t**2 +   t**3) * xx_b*ds
        xp_b[i]   = xp_b[i]   + (    -   t**2 +   t**3) * xx_b*ds
        ss_b = ss_b + ( (  - 6*t + 6*t**2) * x[i-1]      \
                      + (    6*t - 6*t**2) * x[i]        \
                      + (1 - 4*t + 3*t**2) * xp[i-1]*ds  \
                      + (  - 2*t + 3*t**2) * xp[i]  *ds ) * xx_b/ds;

    return x_b, xp_b, ss_b
