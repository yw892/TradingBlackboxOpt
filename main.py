import optunity as op
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization

class OptLearn:
    def __init__(self, obj_fun, pred_fun, param_range):
        self.obj_fun = obj_fun
        self.pred_fun = pred_fun
        self.param_range = param_range

    def fit(self, obj_fun=None, param_range=None,
            num_evals=1000, solver_name="sobol", maximize=True, num_jobs=1, engine="optunity",
            random_state=1, init_points=20, n_iter=50, acq='ucb',
            kappa=2.576, xi=0.0):
        if engine == "optunity":
            if obj_fun:
                self.obj_fun = obj_fun
            if param_range:
                self.param_range = param_range
            if num_jobs == 1:
                if maximize:
                    self.optimal_params, self.info, _ = op.maximize(self.obj_fun,
                                                                    num_evals=num_evals,
                                                                    solver_name=solver_name,
                                                                    **self.param_range)  # default: 'particle swarm'
                else:
                    self.optimal_params, self.info, _ = op.minimize(self.obj_fun,
                                                                    num_evals=num_evals,
                                                                    solver_name=solver_name,
                                                                    **self.param_range)  # de
#                 params
#                 print(self.optimal_params)
#                 optimum and states
                print(self.info.optimum, self.info.stats)
            elif num_jobs > 1:
                if maximize:
                    self.optimal_params, self.info, _ = op.maximize(self.obj_fun,
                                                                    num_evals=num_evals,
                                                                    solver_name=solver_name,
                                                                    pmap=op.parallel.create_pmap(num_jobs),
                                                                    **self.param_range)  # default: 'particle swarm'
                else:
                    self.optimal_params, self.info, _ = op.minimize(self.obj_fun,
                                                                    num_evals=num_evals,
                                                                    solver_name=solver_name,
                                                                    pmap=op.parallel.create_pmap(num_jobs),
                                                                    **self.param_range)  # de
                # params
#                 print(self.optimal_params)
                # optimum and states
                print(self.info.optimum, self.info.stats)
        elif engine == "bayes_opt":
            if obj_fun:
                self.obj_fun = obj_fun
            if param_range:
                self.param_range = param_range
            param_range_t = {}
            for key in self.param_range:
                param_range_t[key] = tuple(self.param_range[key])
            optimizer = BayesianOptimization(
                f=self.obj_fun,
                pbounds=param_range_t,
                random_state=random_state,
            )
            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
                acq=acq,
                kappa=kappa,
                xi=xi,
            )
            self.optimal_params = optimizer.max['params']

    def predict(self, X):
        return self.pred_fun(X, **self.optimal_params)

def create_objective_function(num_w=8):
    def func(*args, **kwargs):
        w=np.array([kwargs['w'+str(i)]+0.00000001 for i in range(1,num_w+1)])
        return sharpe_ratio(w, factors, ret)
    return func

def create_prediction_function(num_w=8):
    def func(X, *args, **kwargs):
        w=np.array([kwargs['w'+str(i)]+0.00000001 for i in range(1,num_w+1)])
        return np.dot(X, w)
    return func

from numba import jit
@jit
def sharpe_ratio(w, factors, ret, num_coins=7):
    signal = np.dot(factors, w)
    profits = np.array([])
    for i in range(0, len(signal) - num_coins, num_coins):
        temp_ret = ret[i:i + num_coins]
        temp_sigal = signal[i:i + num_coins]
        long_ret = np.mean(temp_ret[temp_sigal == np.max(temp_sigal)])
        short_ret = np.mean(temp_ret[temp_sigal == np.min(temp_sigal)])
        profits = np.append(profits, long_ret - short_ret)
    return np.mean(profits) / np.std(profits)

