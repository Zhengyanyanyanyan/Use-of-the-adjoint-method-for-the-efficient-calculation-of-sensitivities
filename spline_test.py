#
# test code to validate spline routines discussed in Section 3.3.3
#

import numpy as np
import spline

# dot product function for vectors and arrays

def dp(A_b,A_d):
    return np.sum(np.multiply(A_b,A_d))

# create random test vectors

N = 6
s    = np.linspace(0.0,1.0,N)  # spline points
x    = np.random.randn(N)
x_d  = np.random.randn(N)
x_b  = np.random.randn(N)
xp_b = np.random.randn(N)

# compute forward and reverse sensitivities for spline creation

xp   = spline.derivs(s, x)
xp_d = spline.derivs(s, x_d)                     # forward mode
x_b2 = spline.derivs_reverse(s, x_b, x_b, xp_b)  # reverse mode

print('spline creation, forward/reverse sensitivity check')
print('adj error: %g \n' % (dp(x_b,x_d) + dp(xp_b,xp_d) - dp(x_b2,x_d)))

# compute forward and reverse sensitivities for spline evaluation

ss   = np.random.rand()  # uniform on (0,1)
ss_d = np.random.randn()
xx_b = np.random.randn()

x_b  = np.zeros_like(x)
xp_b = np.zeros_like(x)
ss_b = 0.

xx_d            = spline.eval_forward(s, x, x_d, xp, xp_d, ss, ss_d)
x_b, xp_b, ss_b = spline.eval_reverse(s, x, x_b, xp, xp_b, ss, ss_b, xx_b)

print('spline evaluation, forward/reverse sensitivity check')
print('adj error: %g \n' % (xx_b*xx_d-ss_b*ss_d-dp(x_b,x_d)-dp(xp_b,xp_d)))
