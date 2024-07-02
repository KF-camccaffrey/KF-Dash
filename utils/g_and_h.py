
"""
FUNCTION      | DESCRIPTION                                  | ARGS
-------------------------------------------------------------------------------------------------------------------
z2gh()        | transform distribution from Z to GH          | z [A, B, g, h]
gh2z()        | transform distribution from GH to Z          | q [A, B, g, h, interval, tol, maxiter, transformed]
deriv_z2gh()  | calculate derivative of z2gh()               | z [B, g, h]
dgh()         | calculate PDF of GH distribution             | x [A, B, g, h, log, interval, tol, maxiter]
rgh()         | randomly sample from a given GH distribution | n [A, B, g, h]
qgh()         | calculate inverse CDF of GH distribution     | p [A, B, g, h]
pgh()         | calculate CDF of GH distribution             | q [A, B, g, h, interval, tol, maxiter]
letterValue() | estimate GH parameters of a distribution     | x [g_, h_, halfspread]
-------------------------------------------------------------------------------------------------------------------
Helper Functions: preprocess(), letterV_Bh_g(), letterV_Bh(), letterV_B(), letterV_g()
"""

# Import Packages
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import median_abs_deviation, norm, linregress
import warnings

DTYPE_INT = np.float64
DTYPE_FLOAT = np.float64

def preprocess(x, params=None):
    x = np.atleast_1d(x).astype(DTYPE_FLOAT)

    if params is None:
        return x

    # initialize result list
    result = [x]

    # x is a 1D vector-like object
    if x.ndim == 1:
        for p in params:
            p = np.atleast_1d(p).astype(DTYPE_FLOAT)
            if len(p) != 1:
                raise ValueError(f"Expected scalar params given x of shape {x.shape}. Actual: {len(p)}")
            result.append(p)
        return tuple(result)

    # x is a 2D matrix-like object
    elif x.ndim == 2:
        for p in params:
            p = np.atleast_1d(p).astype(DTYPE_FLOAT)
            if len(p) != x.shape[0]:
                if len(p) == 1:
                    p = np.tile(p, x.shape[0])
                    warnings.warn(f"Scalar params tiled to length {x.shape[0]} given x of shape {x.shape}", UserWarning)
                else:
                    raise ValueError(f"Expected params of length {x.shape[0]} given x of shape {x.shape}. Actual: {len(p)}")
            result.append(p)
        return tuple(result)

    # x is some other shape
    else:
        raise ValueError(f'x must be a vector-like or matrix-like object')




def z2gh(z, A=0, B=1, g=0, h=0):
    z, A, B, g, h = preprocess(z, [A, B, g, h])
    #z = np.atleast_1d(z).astype(DTYPE_FLOAT)
    #A = np.atleast_1d(A).astype(DTYPE_FLOAT)
    #B = np.atleast_1d(B).astype(DTYPE_FLOAT)
    #g = np.atleast_1d(g).astype(DTYPE_FLOAT)
    #h = np.atleast_1d(h).astype(DTYPE_FLOAT)

    nA = len(A)
    nB = len(B)
    ng = len(g)
    nh = len(h)

    if nA == nB == ng == nh == 1:
        if g == 0:
            print("g1: False")
            termG = np.copy(z)
        else:
            print("g1: True")
            termG = (np.exp(g * z) - 1) / g

        if h <= 0:
            print("h1: False")
            termH = 1
        else:
            print("h1: True")
            termH = np.exp(h * np.power(z, 2) / 2)

        print(f"termG:\n{termG}")
        print(f"termH:\n{termH}")

        return A + B * termG * termH

    elif (z.shape[0] == nA == nB == ng == nh) and (z.ndim == 2):
        termG = np.copy(z)
        termH = np.ones_like(z)
        g1 = (g != 0)
        h1 = (h > 0)

        termG[g1, :] = (np.exp(g[g1].reshape(-1, 1) * z[g1, :]) - 1) / g[g1].reshape(-1, 1)
        termH[h1, :] = np.exp(h[h1].reshape(-1, 1) * np.power(z[h1, :], 2) / 2)

        return A.reshape(-1, 1) + B.reshape(-1, 1) * termG * termH

    else:
        raise ValueError(f'input size mismatch')


def gh2z(q, A=0, B=1, g=0, h=0, interval=(-15, 15), tol=1e-10, maxiter=1000, transformed=False):
    q, A, B, g, h = preprocess(q, [A, B, g, h])
    #q = np.atleast_1d(q).astype(DTYPE_FLOAT)
    #A = np.atleast_1d(A).astype(DTYPE_FLOAT)
    #B = np.atleast_1d(B).astype(DTYPE_FLOAT)
    #g = np.atleast_1d(g).astype(DTYPE_FLOAT)
    #h = np.atleast_1d(h).astype(DTYPE_FLOAT)

    q0 = (q - A) / B if not transformed else q

    g0 = (g == 0)
    h0 = (h == 0)

    out = np.array(q0)  # Ensure q0 is a numpy array for element-wise operations
    interval = np.array(interval)

    if not g0.all() and not h0.all():
        def f(z, i):
            return np.expm1(g[i] * z) * np.exp(h[i] * z**2 / 2)

        if g.size == q0.size:
            for i in range(len(out)):
                out[i] = brentq(lambda z: f(z, i) - q0[i] * g[i], a=interval[0], b=interval[1], xtol=tol, maxiter=maxiter)

        elif g.size == 1:
            for i in range(len(out)):
                out[i] = brentq(lambda z: f(z, 0) - q0[i] * g[0], a=interval[0], b=interval[1], xtol=tol, maxiter=maxiter)
        else:
            raise ValueError(f'input size mismatch')

        return out

    elif g0.all() and h0.all():
        return q0

    elif not g0.all() and h0.all():
        egz = q0 * g + 1
        id = egz <= 0
        if np.any(id):
            out[id] = np.where(g < 0, np.inf, -np.inf)
        out[~id] = np.log(egz[~id]) / g

        return out

    elif g0.all() and not h0.all():
        def f(z):
            return z * np.exp(h * z**2 / 2)

        for i in range(len(out)):
            out[i] = brentq(lambda z: f(z) - q0[i], interval[0], interval[1], xtol=tol, maxiter=maxiter)

        return out

def deriv_z2gh(z, B=1, g=0, h=0):
    z, B, g, h = preprocess(z, [B, g, h])
    #z = np.atleast_1d(z).astype(DTYPE_FLOAT)
    #B = np.atleast_1d(B).astype(DTYPE_FLOAT)
    #g = np.atleast_1d(g).astype(DTYPE_FLOAT)
    #h = np.atleast_1d(h).astype(DTYPE_FLOAT)

    nB = len(B)
    ng = len(g)
    nh = len(h)

    if nB == ng == nh == 1:
        hz2 = h * np.power(z, 2)
        if g == 0:
            trm2 = 1 + hz2

        else:
            e_gz = np.exp(g * z)
            trm2 = e_gz + h * z * (e_gz - 1) / g

        return B * np.exp(hz2 / 2) * trm2

    elif (z.ndim == 2) and (z.shape[0] == nB == ng == nh):
        hz2 = h.reshape(-1, 1) * np.power(z, 2)
        g1 = (g != 0)
        z_g1 = z[g1]

        e_gz1 = np.exp(g[g1].reshape(-1, 1) * z_g1)

        trm2 = np.ones_like(z) + hz2  # Initialize trm2 with 1 + hz2
        trm2[g1] = e_gz1 + h[g1].reshape(-1,1) * z_g1 * (e_gz1 - 1) / g[g1].reshape(-1,1)

        return B.reshape(-1, 1) * np.exp(hz2 / 2) * trm2

    else:
        raise ValueError('length of parameters must match')



def dgh(x, A=0, B=1, g=0, h=0, log=False, interval=(-50, 50), tol=1e-10, maxiter=1000):
    x, A, B, g, h = preprocess(x, [A, B, g, h])
    #x = np.atleast_1d(x).astype(DTYPE_FLOAT)
    #A = np.atleast_1d(A).astype(DTYPE_FLOAT)
    #B = np.atleast_1d(B).astype(DTYPE_FLOAT)
    #g = np.atleast_1d(g).astype(DTYPE_FLOAT)
    #h = np.atleast_1d(h).astype(DTYPE_FLOAT)

    nA = len(A)
    nB = len(B)
    ng = len(g)
    nh = len(h)

    if nA == nB == ng == nh == 1:
        z = np.array(x)
        if isinstance(z, np.ndarray):
            z = z.flatten()
        if h[0] < 0 or B[0] < 0:  # Access elements of lists for scalar checks
            z[...] = np.nan
            return z

        z = q0 = (x - A) / B
        xok = np.isfinite(x)
        z[xok] = gh2z(q=q0[xok], g=g[0], h=h[0], interval=interval, tol=tol, maxiter=maxiter, transformed=True)

    elif (x.ndim == 2) and (x.shape[0] == nA == nB == ng == nh):
        z = q0 = (x - A.reshape(-1, 1)) / B.reshape(-1,1)
        qok = np.isfinite(q0)

        for i in range(nA):
            iok = qok[i, :]
            z[i, iok] = gh2z(q=q0[i, iok], g=g[i], h=h[i], interval=interval, tol=tol, maxiter=maxiter, transformed=True)

    else:
        raise ValueError('length of parameters must match')

    z[np.logical_and(np.isinf(z), z < 0)] = interval[0]
    z[np.logical_and(np.isinf(z), z > 0)] = interval[1]

    deriv = deriv_z2gh(z, B=np.array(B), g=np.array(g), h=np.array(h))

    ret_log = -z**2 / 2 - np.log(2 * np.pi) / 2 - np.log(deriv)
    if log:
        return ret_log
    return np.exp(ret_log)

def rgh(n, A=0, B=1, g=0, h=0):
    z = np.random.normal(size=n)
    return z2gh(z, A, B, g, h)

def qgh(p, A=0, B=1, g=0, h=0):
    z = norm.ppf(p, 0, 1)
    return z2gh(z, A, B, g, h)

def pgh(q, A=0, B=1, g=0, h=0, interval=(-15, 15), tol=1e-10, maxiter=1000):
    z = gh2z(q, A, B, g, h, interval, tol, maxiter)
    return norm.cdf(z, 0, 1)


def letterValue(x, g_=None, h_=None, halfSpread=1):
    # halfspread: 1 = both; 2 = lower; 3 = upper
    x = preprocess(x)
   #x = np.atleast_1d(x).astype(DTYPE_FLOAT)

    if g_ is None:
        g_=np.arange(0.15, 0.255, 0.005)
    if h_ is None:
        h_=np.arange(0.15, 0.355, 0.005)

    if np.any(np.isnan(x)):
        raise ValueError('do not allow NA in observations')

    A = np.median(x)

    if g_ is not False:
        g_ = np.sort(np.unique(g_).astype(DTYPE_FLOAT))
        if len(g_) == 0:
            raise ValueError('`g_` cannot be len-0')
        if np.any((g_ <= 0) | (g_ >= 0.5)):
            raise ValueError('g_ (for estimating g) must be between 0 and .5 (not including)')
        L = A - np.quantile(x, q=g_)
        U = np.quantile(x, q=1 - g_) - A
        ok = (L != 0) & (U != 0)
        if not np.all(ok):
            g_ = g_[ok]
        if len(g_) == 0:
            raise ValueError('`g_` cannot be len-0')

    if h_ is not False:
        h_ = np.sort(np.unique(h_).astype(DTYPE_FLOAT))
        if len(h_) == 0:
            raise ValueError('`h_` cannot be len-0')
        if np.any((h_ <= 0) | (h_ >= 0.5)):
            raise ValueError('h_ (for estimating h) must be between 0 and .5 (not including)')
        L = A - np.quantile(x, q=h_)
        U = np.quantile(x, q=1 - h_) - A
        ok = (L != 0) & (U != 0)
        if not np.all(ok):
            h_ = h_[ok]
        if len(h_) == 0:
            raise ValueError('`h_` cannot be len-0')

    #halfSpread = halfSpread.lower()

    if g_ is False:
        g = 0
    else:
        g = letterV_g(x, A, g_)

    if h_ is False:
        if g == 0:
            return {'A': A, 'B': median_abs_deviation(x, center=A), 'g': g, 'h': 0}
        B = letterV_B(x, A, g, g_)
        ret = {'A': A, 'B': B[halfSpread], 'g': g, 'h': 0}
        ret['B'] = B
    else:
        if g != 0:
            Bh = letterV_Bh_g(x, A, g, h_)
        else:
            Bh = letterV_Bh(x, A, h_)
        ret = {'A': A, 'B': Bh['B'][halfSpread], 'g': g, 'h': Bh['h'][halfSpread]}
        ret['Bh'] = Bh

    ret['g'] = g
    ret['h'] = Bh['h'] if 'Bh' in ret else 0

    return ret

def letterV_Bh_g(x, A, g, h_):
    L = A - np.quantile(x, q=h_)
    U = np.quantile(x, q=1 - h_) - A
    z = norm.ppf(h_)
    regx = z**2 / 2

    regyU = np.log(g * U / np.expm1(-g*z))
    regyL = np.log(g * L / (-np.expm1(g*z)))

    regy = np.log(g * (U+L) / (np.exp(-g*z) - np.exp(g*z)))

    X = np.vstack((np.ones_like(regx), regx)).T
    cf = linregress(regx, regy)
    cfL = linregress(regx, regyL)
    cfU = linregress(regx, regyU)

    B = np.exp([cf.intercept, cfL.intercept, cfU.intercept])
    h = np.maximum(0, [cf.slope, cfL.slope, cfU.slope])

    return {'B': B, 'h': h}

def letterV_Bh(x, A, h_):
    L = A - np.quantile(x, q=h_)
    U = np.quantile(x, q=1 - h_) - A
    z = norm.ppf(h_)
    regx = z**2 / 2

    regy = np.log((U+L) / (-2*z))
    regyL = np.log(-L/z)
    regyU = np.log(-U/z)

    X = np.vstack((np.ones_like(regx), regx)).T
    cf = linregress(regx, regy)
    cfL = linregress(regx, regyL)
    cfU = linregress(regx, regyU)

    B = np.exp([cf.intercept, cfL.intercept, cfU.intercept])
    h = np.maximum(0, [cf.slope, cfL.slope, cfU.slope])

    return {'B': B, 'h': h}

def letterV_B(x, A, g, g_):
    q = np.quantile(x, q=[g_, 1-g_])
    z = norm.ppf(g_)
    regx = np.expm1(g*z) / g

    B = minimize_scalar(lambda B: np.sum((q - A - B*regx)**2), bounds=(0, None)).x
    BL = minimize_scalar(lambda BL: np.sum((q[0] - A - BL*regx[0])**2), bounds=(0, None)).x
    BU = minimize_scalar(lambda BU: np.sum((q[1] - A - BU*regx[1])**2), bounds=(0, None)).x

    return {'both': B, 'lower': BL, 'upper': BU}

def letterV_g(x, A, g_):
    L = A - np.quantile(x, q=g_)
    U = np.quantile(x, q=1 - g_) - A
    gs = np.log(L / U) / norm.ppf(g_)
    g = np.median(gs)

    return g

if __name__ == "__main__":
    print("g_and_h testing")

    x = np.linspace(-10, 10, 500)
    xx = np.tile(x, (3,1))

    g = [-0.5, 0, 0.5]
    h = 0.05

    A = [-5, 0, 5]

    y = dgh(xx, A=A, B=1, g=g, h=h)

    import matplotlib.pyplot as plt

    plt.scatter(xx.flatten(), y.flatten())
    plt.show()
