try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp

import numpy as np

from nda.optimizers.utils import GD, FISTA, NAG
from nda.optimizers import Optimizer

def bfgs_update(bfgs_matrix, s, y,inv = False):
    '''Based on Nocedal et al, Numerical Optimization book, 2006
       s = x(t + 1) - x(t)
       y = grad(t + 1) - grad (t)'''

    # Normalization to increase numerical stability
    s = s/xp.linalg.norm(y)
    y = y/xp.linalg.norm(y)

    if not inv:
        ### BFGS update
        ## Auxiliary computation
        # B(t)*s(t)
        B_prod = bfgs_matrix @ s
        # B(t)*s(t)s^T(t)B^T(t)
        sub_B = xp.outer(B_prod,B_prod)
        # s(t)^TB(t)s(t)
        s_inner_B = xp.dot(s,B_prod)
        # y outer product
        y_out = xp.outer(y,y)
        # y outer rescale
        y_scale = xp.dot(y,s)
        # print(f'<y,s> = {y_scale}')
        if y_scale <10**-5:#  Works with 10**-3

            return bfgs_matrix

        BFGS_up = bfgs_matrix - sub_B/s_inner_B + y_out/y_scale

        return BFGS_up

    else:
        ### BFGS inverse update
        ## Auxiliary computation
        # H(t)*y(t)
        H_prod = bfgs_matrix @ y
        # y(t)^TH(t)y(t)
        y_inner_H = xp.dot(y,H_prod)
        # rho
        y_scale = xp.dot(y,s)
        if y_scale < 10**-5:#== 0:

            return bfgs_matrix
        rho = 1/y_scale
        # s(t)s(t)^T
        s_outer = xp.outer(s,s)
        # y(t)s(t)^T
        #y_outer_s = xp.outer(y,s)

        BFGS_inv_up = bfgs_matrix - rho*(xp.outer(s,H_prod) + xp.outer(H_prod,s) - (rho*y_inner_H + 1)*s_outer)

        return BFGS_inv_up


class BFGS(Optimizer):
    '''This is a BFGS implementation for decentralized optimization. It does not perform line-search, i.e. it uses a fixed step size.'''

    def __init__(self, p, mu, local_n_iters=200, **kwargs):
        super().__init__(p, **kwargs)
        self.mu = mu
        self.local_n_iters = local_n_iters
        self.B = np.identity(self.p.dim)
        self.v = np.zeros((self.p.dim,))
        self.a = np.zeros((self.p.dim,))


    def update(self):
        self.comm_rounds += 1
        
        grad_x = self.grad(self.x)

        x_last = self.x.copy()
        grad_x_last = grad_x.copy()

        x_next = x_last - self.mu*self.B@grad_x_last

        self.x = x_next

        grad_x_last = grad_x.copy()
        grad_x = self.grad(self.x)

        self.v = self.x - x_last
        self.a= grad_x - grad_x_last

        self.B = bfgs_update(self.B,self.v,self.a,inv=True)


class CEDANE(Optimizer):
    '''CEDANE algorithm described in "Harvesting Curvatures for Communication-Efficient Distributed Optimization", 
        Asilomar Conference on Signals, Systems and Computers 2022.'''

    def __init__(self, p, mu, local_n_iters=100, local_optimizer='NAG', delta=None, **kwargs):
        super().__init__(p, **kwargs)
        self.mu = mu
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.delta = delta

        self.v = np.zeros((self.p.dim,))
        self.a = np.zeros((self.p.dim,))
        self.B = np.identity(self.p.dim)

        
    def update(self):
        self.comm_rounds += 2

        grad_x = self.grad_h(self.x)

        x_next = 0
        for i in range(self.p.n_agent):

            if self.p.is_smooth is False:
                grad_x_i = self.grad_h(self.x, i)

                def _grad(tmp):
                    return self.grad_h(tmp, i) - grad_x_i + grad_x + self.mu * self.B @ (tmp - self.x)
                tmp, count = FISTA(_grad, self.x.copy(), self.mu + 1, self.p.r, n_iters=self.local_n_iters, eps=1e-10)

            else:
                grad_x_i = self.grad_h(self.x, i)

                def _grad(tmp):
                    return self.grad_h(tmp, i) - grad_x_i + grad_x + self.mu * self.B @ (tmp - self.x)

                if self.local_optimizer == "NAG":
                    if self.delta is not None:
                        tmp, count_ = NAG(_grad, self.x.copy(), self.delta, self.local_n_iters, eps=self.var_eps)
                    else:
                        tmp, count_ = NAG(_grad, self.x.copy(), self.p.L + self.mu, self.p.sigma + self.mu, self.local_n_iters, eps=self.var_eps)

                else:
                    if self.delta is not None:
                        tmp, count_ = GD(_grad, self.x.copy(), self.delta, self.local_n_iters)
                    else:
                        tmp, count_ = GD(_grad, self.x.copy(), 2 / (self.p.L + self.mu + self.p.sigma + self.mu), self.local_n_iters)

            x_next += tmp

        x_last = self.x.copy()

        self.x = x_next / self.p.n_agent
        grad_x_last = grad_x.copy()
        grad_x = self.grad_h(self.x)

        self.v = self.x - x_last
        self.a = grad_x - grad_x_last
        
        # All agents are computing the same matrix, so this only needs to be computed once in the simulations
        self.B = bfgs_update(self.B,self.v,self.a,inv=False)
