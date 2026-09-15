#!/usr/bin/env python3
"""Dependency-free asymptotic expected CLs limits for simplified HH4b models."""
import math

def _q(mu,s,b,v):
    if s<0 or b<=0 or v<0: raise ValueError("require s>=0, b>0, v>=0")
    if not s: return 0.0
    n=b
    if not v:
        lam=mu*s+b
        return 2*(lam-n+n*math.log(n/lam))
    a=v-mu*s-b
    lam=.5*(-a+math.sqrt(a*a+4*n*v))
    bhat=lam-mu*s
    lr=n*math.log(lam/n)-(lam-n)-(bhat-b)**2/(2*v)
    return max(0.,-2*lr)

def qmu_asimov(mu,signal,background,background_variance=None):
    if len(signal)!=len(background) or not signal: raise ValueError("length mismatch")
    variance=[0.]*len(signal) if background_variance is None else background_variance
    if len(variance)!=len(signal): raise ValueError("variance length mismatch")
    return sum(_q(mu,s,b,v) for s,b,v in zip(signal,background,variance))

def expected_upper_limit(signal,background,background_variance=None,confidence_level=.95):
    """Median b-only asymptotic CLs limit; CLs=2[1-Phi(sqrt(q_mu))]."""
    if confidence_level != .95: raise ValueError("only 95% CL is frozen")
    target=1.959963984540054**2
    lo,hi=0.,1.
    while qmu_asimov(hi,signal,background,background_variance)<target:
        hi*=2
        if hi>1e15: raise RuntimeError("cannot bracket")
    for _ in range(100):
        mid=(lo+hi)/2
        if qmu_asimov(mid,signal,background,background_variance)<target: lo=mid
        else: hi=mid
    return (lo+hi)/2

def _q_gamma(mu,s,b,v):
    """Poisson-equivalent gamma constraint with tau=b^2/v effective counts."""
    if not v: return _q(mu,s,b,0.)
    tau=b*b/v
    if tau<=0 or b<=0: raise ValueError("gamma model requires positive b and variance")
    n=b
    def score(beta): return b*n/(mu*s+beta*b)-b+tau/beta-tau
    lo,hi=1e-14,1.
    while score(hi)>0: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if score(mid)>0: lo=mid
        else: hi=mid
    beta=(lo+hi)/2; lam=mu*s+beta*b
    lr=n*math.log(lam/n)-(lam-n)+tau*math.log(beta)-(beta*tau-tau)
    return max(0.,-2*lr)

def expected_upper_limit_gamma(signal,background,background_variance):
    """Median 95% CLs limit using positive-bin effective-count constraints."""
    if not (len(signal)==len(background)==len(background_variance)) or not signal: raise ValueError("length mismatch")
    target=1.959963984540054**2
    def q(mu): return sum(_q_gamma(mu,s,b,v) for s,b,v in zip(signal,background,background_variance))
    lo,hi=0.,1.
    while q(hi)<target:
        hi*=2
        if hi>1e15: raise RuntimeError("cannot bracket")
    for _ in range(100):
        mid=(lo+hi)/2
        if q(mid)<target: lo=mid
        else: hi=mid
    return (lo+hi)/2
