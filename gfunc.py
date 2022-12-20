import numpy as np
import scipy.optimize as op

class Gfunc(object):
    def __init__(self, gfunc, dgdq=None, probdata=None, analysisopt=None,
            dgthetag=None, evaluator='basic', param=None, gftype='userfunc'):
        self.evaluator = evaluator
        self.param = param
        self.gftype = gftype
        self.dgdq = dgdq
        self.dgthetag = dgthetag
        self.gfunc = gfunc
        self.probdata = probdata
        self.analysisopt = analysisopt

    def setparam(self, param):
        self.param = param

    def g_value(self, x, param=None):
        if self.gftype == 'userfunc':
            if param is None:
                return self.gfunc(x)
            else:
                return self.gfunc(x, param)
        else:
            return None

    def dgdq_value(self, x, param=None):
        if self.gftype == 'userfunc':
            if self.dgdq is None:
                assert self.probdata is not None, "Gfunc with FFD needs probdata"
                stds = np.array( [np.sqrt(rv.stats('v')[()]) for rv in self.probdata.rvs] )
                ffdparam, ffdparamthetag = self.analysisopt.ffdparam, self.analysisopt.ffdparamthetag
                ffdstep = stds/ffdparam
                if param is None:
                    dgdq = op.approx_fprime(x, self.gfunc, ffdstep)
                else:
                    ffdstepthetag = param/ffdparamthetag
                    g0 = self.gfunc(x, param)
                    g1 = self.gfunc(x+ffdstep, param+ffdstepthetag)
                    dgdq = (g1-g0) / np.hstack((ffdstep,ffdstepthetag))
                return dgdq
            else:
                if param is None:
                    return self.dgdq(x, self.param)
                else:
                    return self.dgdq(x, param)
        else:
            return None


if __name__ == '__main__':
    def gf1(x, param=None):
        return x[0]-x[1]
    def dg1dq(x, param=None):
        return [1.,-1.]
    gfunc = Gfunc(gf1, dgdq=dg1dq)
    print(  (gfunc.g_value([3,1]), gfunc.dgdq_vale([3,1])) )
