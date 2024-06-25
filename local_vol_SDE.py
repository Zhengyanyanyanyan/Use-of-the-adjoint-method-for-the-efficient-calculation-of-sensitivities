import numpy as np
import spline

# dot product function for vectors and arrays

def dp(A_b,A_d):
    return np.sum(np.multiply(A_b,A_d))

# spline data

s   = np.array([ 0  , 1  , 2   ])
sig = np.array([ 0.2, 0.4, 0.2 ])

T  = 1
N  = 10            # number of timesteps
h  = T/N;  sqrt_h = np.sqrt(h)
r  = 0.05
S0 = 1.0
K  = 0.5

dW = np.sqrt(h)*np.random.randn(N)

# forward mode

sig_d  = np.random.randn(s.size)
sigp   = spline.derivs(s, sig)
sigp_d = spline.derivs(s, sig_d)

S = S0;  S_d = np.zeros_like(S)

for n in range(0,N):
    sigma   = spline.eval(s,sig,sigp, S)
    sigma_d = spline.eval_forward(s, sig,sig_d, sigp,sigp_d, S,S_d)
    S_d = S_d*( 1 + r*h + sigma*dW[n] ) + S*sigma_d*dW[n]
    S   = S  *( 1 + r*h + sigma*dW[n] )

P   = np.exp(-r*T)*np.maximum(0,S-K);
P_d = np.exp(-r*T)*np.heaviside(S-K,0.5)*S_d;

print('forward mode sensitivity:  %f' % P_d)

# reverse mode

S_copy = np.zeros(N)
S = S0;

for n in range(0,N):
    S_copy[n] = S
    sigma = spline.eval(s,sig,sigp, S)
    S = S*( 1 + r*h + sigma*dW[n] )

S_b    = np.exp(-r*T)*np.heaviside(S-K,0.5)
sig_b  = np.zeros_like(sig)
sigp_b = np.zeros_like(sig)

for n in range(N-1,-1,-1):
    S = S_copy[n]
    sigma = spline.eval(s,sig,sigp, S)
    sigma_b = S_b*S*dW[n]
    S_b = S_b*( 1 + r*h + sigma*dW[n] )
    sig_b, sigp_b, S_b = \
        spline.eval_reverse(s, sig,sig_b, sigp,sigp_b, S,S_b, sigma_b)

print('intermediate check:        %f' % (dp(sig_b,sig_d) + dp(sigp_b,sigp_d)))

sig_b = spline.derivs_reverse(s, sig, sig_b, sigp_b)  #why need this

print('reverse mode sensitivity:  %f' % dp(sig_b,sig_d) )



 




  

  
