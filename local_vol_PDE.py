import numpy as np
import matplotlib.pyplot as plt
import call_options as calls
import trid
import spline

# dot product function for vectors and arrays

def dp(A_b,A_d):
    return np.sum(np.multiply(A_b,A_d))

#
# define BS discretisation with local volatility
#


def BS_discretisation(J,dt,dS, r, s,sig,sigp):
    a = np.zeros(J+1);  b = np.zeros(J+1);  c = np.zeros(J+1)
    
    for j in range(J):
        sigma = spline.eval(s,sig,sigp, j*dS)
        a[j] =  dt*( 0.5*sigma**2*j*j - 0.5*r*j     )
        b[j] =  dt*(   - sigma**2*j*j           - r )
        c[j] =  dt*( 0.5*sigma**2*j*j + 0.5*r*j     )

    a[J] = dt*( -r*J     )   # enforce right-hand b.c.
    b[J] = dt*(  r*J - r )

    return trid.assemble(a,b,c)

def BS_discretisation_forward(J,dt,dS, r,r_d, s,sig,sig_d,sigp,sigp_d):
    a_d = np.zeros(J+1); b_d = np.zeros(J+1); c_d = np.zeros(J+1)
    
    for j in range(J):
        sigma   = spline.eval(s,sig,  sigp,   j*dS)
        sigma_d = spline.eval(s,sig_d,sigp_d, j*dS)
        a_d[j] =  dt*(     sigma*sigma_d*j*j - 0.5*r_d*j     )
        b_d[j] =  dt*(-2.0*sigma*sigma_d*j*j           - r_d )
        c_d[j] =  dt*(     sigma*sigma_d*j*j + 0.5*r_d*j     )

    a_d[J] = dt*( -r_d*J       )   # enforce right-hand b.c.
    b_d[J] = dt*(  r_d*J - r_d )

    return trid.assemble(a_d,b_d,c_d)

def BS_discretisation_reverse(J,dt,dS, r,r_b, s,sig,sig_b,sigp,sigp_b, D_b):
    a_b, b_b, c_b = trid.disassemble(D_b)
    
    for j in range(J):
        sigma   = spline.eval(s,sig,sigp, j*dS)
        sigma_b = dt*sigma*j*j*( a_b[j] - 2.0*b_b[j] + c_b[j] )
        ss_b = 0.
        sig_b, sigp_b, ss_b = \
            spline.eval_reverse(s,sig,sig_b,sigp,sigp_b, j*dS, ss_b, sigma_b)
        r_b = r_b + dt*( - 0.5*j*a_b[j] - b_b[j] + 0.5*j*c_b[j] )

    r_b = r_b + dt*( - J*a_b[J] + (J - 1.0)*b_b[J] )

    return r_b, sig_b, sigp_b

#
# various initialisations
#

plt.ion()

r  = 0.05
T  = 1   # maturity
S0 = 100 # S_0
K  = 100 # strike

M = 10
s    = np.linspace(0,2*S0,M)
sig  = np.linspace(0.2,0.4,M)
sigp = spline.derivs(s,sig)

# section to validate BS_discretisation functions

r_d    = np.random.randn()
sig_d  = np.random.randn(M)
sigp_d = spline.derivs(s,sig_d)

J  = 64
N  = 16
dS = 2*S0/J
dt = T/N

D_d = BS_discretisation_forward(J,dt,dS, r,r_d, s,sig,sig_d,sigp,sigp_d)

D_b = np.random.randn(3,J+1)
D_b[0,0] = 0; D_b[-1,-1] = 0

print('forward sensitivity %f ' % dp(D_b, D_d))

r_b = 0.
sig_b  = np.zeros_like(sig)
sigp_b = np.zeros_like(sig)

r_b, sig_b, sigp_b = \
  BS_discretisation_reverse(J,dt,dS, r,r_b, s,sig,sig_b,sigp,sigp_b, D_b)

print('reverse sensitivity %f ' % (r_b*r_d+dp(sig_b,sig_d)+dp(sigp_b,sigp_d)))

# "forward" mode  (actually backward in time)

S = np.linspace(0,2*S0,J+1) #  np.arange(J+1)*dS
u = np.maximum(S-K,0)

u_d = np.random.randn(J+1)
u_d_copy = u_d.copy()

D   = BS_discretisation(J,dt,dS, r, s,sig,sigp)
D_d = BS_discretisation_forward(J,dt,dS, r,r_d, s,sig,sig_d,sigp,sigp_d)

I   = trid.identity(J+1)
A   = I - 0.5*D;    B   = I + 0.5*D
A_d =   - 0.5*D_d;  B_d =   + 0.5*D_d

u_copy = np.zeros((J+1,N+1))  # save for reverse mode
u_copy[:,N] = u

for n in range(N-1,-1,-1):
    v   = trid.multiply(B,u)
    v_d = trid.multiply(B,u_d) + trid.multiply(B_d,u)
    u   = trid.solve(A, v)
    u_d = trid.solve(A, v_d - trid.multiply(A_d,u))
    u_copy[:,n] = u

P   = u[J//2]
P_d = u_d[J//2]

print('\nforward sensitivity %f ' % P_d)

# "reverse" mode  (actually forward in time)

u_b = np.zeros_like(u)
u_b[J//2] = 1.

A_b = np.zeros_like(A)
B_b = np.zeros_like(B)
D_b = np.zeros_like(D)

u = u_copy[:,0]

for n in range(0,N):
    v_b = trid.solve(trid.transpose(A), u_b)
    A_b = A_b - trid.outer(v_b,u)
    u = u_copy[:,n+1]
    B_b = B_b + trid.outer(v_b,u)
    u_b = trid.multiply(trid.transpose(B), v_b)

D_b = - 0.5*A_b + 0.5*B_b
r_b = 0.
sig_b  = np.zeros_like(sig)
sigp_b = np.zeros_like(sig)

r_b, sig_b, sigp_b = \
  BS_discretisation_reverse(J,dt,dS, r,r_b, s,sig,sig_b,sigp,sigp_b, D_b)

print('reverse sensitivity %f ' % \
      (dp(u_b,u_d_copy) + r_b*r_d + dp(sig_b,sig_d) + dp(sigp_b,sigp_d)))
