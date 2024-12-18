from firedrake import Function
from gusto.core.fields import Fields

import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

class PararealFields(object):

    def __init__(self, equation, nlevels):
        levels = [str(n) for n in range(nlevels+1)]
        self.add_fields(equation, levels)

    def add_fields(self, equation, levels):
        if levels is None:
            levels = self.levels
        for level in levels:
            try:
                x = getattr(self, level)
                x.add_field(equation.field_name, equation.function_space)
            except AttributeError:
                setattr(self, level, Fields(equation))

    def __call__(self, n):
        return getattr(self, str(n))


class PararealAbstr(object):

    def __init__(self, domain, coarse_scheme, fine_scheme, nG, nF,
                 n_intervals, max_its):

        assert coarse_scheme.nlevels == 1
        assert fine_scheme.nlevels == 1
        self.nlevels = 1

        self.coarse_scheme = coarse_scheme
        self.coarse_scheme.dt.assign(domain.dt/n_intervals/nG)
        self.fine_scheme = fine_scheme
        self.fine_scheme.dt.assign(domain.dt/n_intervals/nG)
        self.nG = nG
        self.nF = nF
        self.n_intervals = n_intervals
        self.max_its = max_its

    def setup(self, equation, apply_bcs=True, *active_labels):
        self.coarse_scheme.fixed_subcycles = self.nG
        self.coarse_scheme.setup(equation, apply_bcs, *active_labels)
        self.fine_scheme.fixed_subcycles = self.nF
        self.fine_scheme.setup(equation, apply_bcs, *active_labels)
        self.x = PararealFields(equation, self.n_intervals)
        self.xF = PararealFields(equation, self.n_intervals)
        self.xn = Function(equation.function_space)
        self.xGk = PararealFields(equation, self.n_intervals)
        self.xGkm1 = PararealFields(equation, self.n_intervals)
        self.xFn = Function(equation.function_space)
        self.xFnp1 = Function(equation.function_space)
        self.name = equation.field_name

    def setup_transporting_velocity(self, uadv):
        self.coarse_scheme.setup_transporting_velocity(uadv)
        self.fine_scheme.setup_transporting_velocity(uadv)

    def apply(self, x_out, x_in):

        self.xn.assign(x_in)
        x0 = self.x(0)(self.name)
        x0.assign(x_in)
        xF0 = self.xF(0)(self.name)
        xF0.assign(x_in)

        # compute first guess from coarse scheme
        for n in range(self.n_intervals):
            print("computing first coarse guess for interval: ", n)
            # apply coarse scheme and save data as initial conditions for fine
            xGnp1 = self.xGkm1(n+1)(self.name)
            self.coarse_scheme.apply(xGnp1, self.xn)
            xnp1 = self.x(n+1)(self.name)
            xnp1.assign(xGnp1)
            self.xn.assign(xnp1)

        for k in range(self.max_its):

            # apply fine scheme in each interval using previously
            # calculated coarse data
            for n in range(k, self.n_intervals):
                print("computing fine guess for iteration and interval: ", k, n)
                self.xFn.assign(self.x(n)(self.name))
                xFnp1 = self.xF(n+1)(self.name)
                self.fine_scheme.apply(xFnp1, self.xFn)

            self.update_dataset(init_cond=self.x,
                                F_init_cond=self.xF,
                                G_init_cond=self.xGkm1,
                                para_iters=range(k, self.n_intervals))

            # compute correction
            for n in range(k, self.n_intervals):
                xn = self.x(n)(self.name)
                xGk = self.xGk(n+1)(self.name)
                # compute new coarse guess
                self.coarse_scheme.apply(xGk, xn)
                xnp1 = self.x(n+1)(self.name)
                xGkm1 = self.xGkm1(n+1)(self.name)
                xFnp1 = self.xF(n+1)(self.name)
                self.apply_update_rule(xnp1, xGk, xFnp1 - xGkm1, xn)
                xGkm1.assign(xGk)

        x_out.assign(xnp1)

    def update_dataset(self, init_cond, F_init_cond, G_init_cond, para_iters):
        '''
        Update dataset using the initial initial conditions from the previous iteration,
        and the application of F and G to them. Specifically, if U_i^k is the initial
        condition at time t_i and iteration k, update with
        U_{i}^{k-1}, F(U_{i}^{k-1}), and G(U_{i}^{k-1}).
        '''
        raise NotImplementedError("update_dataset method must be implemented in subclass")

    def apply_update_rule(self, xnp1, xGk, correction, new_init_cond):
        '''
        Calculate the update rule for the next iteration.
        '''
        raise NotImplementedError("calc_update_rule method must be implemented in subclass")
    
class Parareal(PararealAbstr):
    def update_dataset(self, *args, **kwargs):
        # We don't need to store any data for Parareal
        pass

    def apply_update_rule(self, xnp1, xGk, correction, new_init_cond):
        xnp1.assign(xGk + correction)
    


class Dataset(object):
    def __init__(self):
        self.dtset_x = None
        self.dtset_y = None
        self.tempx = []
        self.tempy = []

    def add_obsv(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        
        self.tempx.append(x.copy())
        self.tempy.append(y.copy())

    def collect(self):
        self.dtset_x = self._to_numpy_and_merge(self.dtset_x, self.tempx)
        self.dtset_y = self._to_numpy_and_merge(self.dtset_y, self.tempy)
        self.tempx = []
        self.tempy = []

    def get_data(self):
        return self.dtset_x, self.dtset_y

    @staticmethod
    def _to_numpy_and_merge(dtset, temp):
        if len(temp) == 0:
            raise ValueError("No data has been added to the dataset")
        
        if dtset is None:
            return np.array(temp)
        else:
            return np.concatenate((dtset, np.array(temp)), axis=0)
        

class RandNet():
    '''
    A Random Neural Network for Local Regression with random polynomial-expanded features.

    This class implements a random neural network that retrains on a subset 
    of the dataset for each prediction. The subset is dynamically selected 
    as the `m` nearest neighbors of the target point `new_x`. 
    '''
    
    def __init__(self, d, seed=47, res_size=100, degree=1, m=10):
        '''
        Initializes a RandNet instance.

        Args:
            d (int): The dimensionality of the input data.
            seed (int, optional): Seed for the random number generator to ensure 
                reproducibility. Default is 47.
            res_size (int, optional): The number of neurons in the hidden layer 
                (reservoir size). Default is 100.
            degree (int, optional): The degree of the polynomial feature expansion. 
                Default is 1 (linear features).
            m (int, optional): The number of nearest neighbors to use for retraining 
                during prediction. Default is 10.
        '''
        self.d = d
        self.N = res_size
        self.rng = np.random.default_rng(seed)
        self.m = m
        
        self.loss = lambda x: np.maximum(x, 0)
        self.M = 1
        self.R = 1
        self.alpha = 0
        self.poly = PolynomialFeatures(degree=degree)
        self.poly.fit(np.zeros((1, d)))
        self.degree = self.poly.n_output_features_
        
        bias, C = self._init_obj()
        self.bias, self.C = bias, C

        
    def _init_obj(self):
        N, rng = self.N, self.rng
        bias = rng.uniform(-1, 1, (N, 1))
        C = rng.uniform(-1, 1, (N, self.degree))
        return bias, C
    
    def _fit(self, x, y, bias, C):
        x = self.poly.fit_transform(x)
        X = self.loss(bias + C @ x.T) # activation
        X = X.T #first col is intercept
        mdl = Ridge(alpha=self.alpha)
        mdl.fit(X, y)
        return mdl
        

    def fit(self, x, y):

        # Normalize data between -1 and 1
        mn = np.min(x, axis=0)
        mx = np.max(x, axis=0)

        self.x = 2*(x-mn)/(mx-mn)-1
        self.y = 2*(y-mn)/(mx-mn)-1
        self.norm_min = mn
        self.norm_max = mx
        

    def predict(self, new_x):
        mn, mx = self.norm_min, self.norm_max
        bias = self.M * self.R * self.bias
        bias = self.bias
        C = self.R * self.C

        # normalize input
        new_x = 2*(new_x-mn)/(mx-mn)-1

        # Compute the nearest neighbors
        s_idx = np.argsort(scipy.spatial.distance.cdist(new_x, self.x, metric='sqeuclidean')[0,:])
        # print('>', s_idx[:self.m])
        xm = self.x[s_idx[:self.m], :]
        ym = self.y[s_idx[:self.m], :]
        
        # Compute input features
        new_X = self.poly.fit_transform(new_x)
        _int = bias + C @ new_X.T
        new_X = self.loss(_int)
        
        # Fit the model and make predictions
        mdl = self._fit(xm, ym, bias, C)
        preds = np.squeeze(mdl.predict(new_X.T))

        # Unnormalize the output
        preds = (preds+1)/2 * (mx-mn) + mn

        return preds
    


class RandNetParareal(Parareal):
    '''
    An implementation of: RandNet-Parareal: a time-parallel PDE solver using Random Neural Networks. 

    Citation:
    @article{gattiglio2024randnet,
    title={RandNet-Parareal: a time-parallel PDE solver using Random Neural Networks},
    author={Gattiglio, Guglielmo and Grigoryeva, Lyudmila and Tamborrino, Massimiliano},
    journal={Advances in neural information processing systems},
    year={2024}
    }

    See https://arxiv.org/pdf/2411.06225v1 for details on the effect of the
    parameters, specifically Appendix D.

    
    '''
    def __init__(self, *args, n_neurons=100, n_neighbors=10, poly_expansion_degree=1, **kwargs):
        '''
        See https://arxiv.org/pdf/2411.06225v1 for details on the effect of the parameters, specifically Appendix D.

        Args:
            *args: Positional arguments passed to the base `Parareal` class.
            n_neurons (int, optional): Number of neurons in the hidden layer 
                of the `RandNet` (reservoir size). Default is 100.
            n_neighbors (int, optional): Number of nearest neighbors used by 
                the `RandNet` during predictions. Default is 10.
            poly_expansion_degree (int, optional): Degree of polynomial 
                expansion for input features in the `RandNet`. Default is 1.
            **kwargs: Additional keyword arguments passed to the base `Parareal` class.
        '''
        super().__init__(*args, **kwargs)

        # Dataset stores for RandNet-Parareal
        self.dtset = Dataset()

        self.n_neurons = n_neurons
        self.n_neighbors = n_neighbors
        self.poly_expansion_degree = poly_expansion_degree

    def setup(self, equation, apply_bcs=True, *active_labels):

        super().setup(equation, apply_bcs, *active_labels)
        state_dimension = len(equation.X.dat.data[:])

        # Random neural network
        self.randnet = RandNet(d=state_dimension, res_size=self.n_neurons, degree=self.poly_expansion_degree, m=self.n_neighbors)

        

    def update_dataset(self, init_cond, F_init_cond, G_init_cond, para_iters):
        
        # store each observation
        for n in para_iters:
            dtset_x = init_cond(n)(self.name).dat.data
            dtset_y = F_init_cond(n+1)(self.name).dat.data - G_init_cond(n+1)(self.name).dat.data
            self.dtset.add_obsv(dtset_x, dtset_y)

        # gather the dataset -> convert to numpy 
        self.dtset.collect()

        # fit the RandNet on the data
        self.randnet.fit(*self.dtset.get_data())

    def apply_update_rule(self, xnp1, xGk, correction, new_init_cond):
        new_init_cond = new_init_cond.dat.data

        pred = self.randnet.predict(new_init_cond.reshape(1, -1))

        xnp1.assign(correction)
        xnp1.dat.data[:] = xGk.dat.data[:] + pred
    







