import numpy as np
from abc import ABCMeta
import types
import warnings
from types import MethodType, FunctionType
from functools import lru_cache

def set_run_mode(func, mode):
    '''
    :param func:
    :param mode: string
        can be  common, vectorization , parallel, cached
    :return:
    '''
    # if mode == 'multiprocessing' and sys.platform == 'win32':
    #     warnings.warn('multiprocessing not support in windows, turning to multithreading')
    #     mode = 'multithreading'
    # if mode == 'parallel':
    #     mode = 'multithreading'
    #     warnings.warn('use multithreading instead of parallel')
    func.__dict__['mode'] = mode
    return


def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    :param func:
    :return:
    '''

    # to support the former version
    if (func.__class__ is FunctionType) and (func.__code__.co_argcount > 1):
        warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

        def func_transformed(X):
            return np.array([func(*tuple(x)) for x in X])

        return func_transformed

    # to support the former version
    if (func.__class__ is MethodType) and (func.__code__.co_argcount > 2):
        warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

        def func_transformed(X):
            return np.array([func(tuple(x)) for x in X])

        return func_transformed

    # to support the former version
    if getattr(func, 'is_vector', False):
        warnings.warn('''
        func.is_vector will be deprecated in the future, use set_run_mode(func, 'vectorization') instead
        ''')
        set_run_mode(func, 'vectorization')

    mode = getattr(func, 'mode', 'others')
    valid_mode = ('common', 'vectorization', 'cached', 'others')
    assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)

    if mode == 'vectorization':
        return func
    elif mode == 'cached':
        @lru_cache(maxsize=None)
        def func_cached(x):
            return func(x)

        def func_warped(X):
            return np.array([func_cached(tuple(x)) for x in X])

        return func_warped
    # elif mode == 'multithreading':
    #     from multiprocessing.dummy import Pool as ThreadPool

    #     pool = ThreadPool()

    #     def func_transformed(X):
    #         return np.array(pool.map(func, X))

    #     return func_transformed
    # elif mode == 'multiprocessing':
    #     from multiprocessing import Pool
    #     pool = Pool()

    #     def func_transformed(X):
    #         return np.array(pool.map(func, X))

    #     return func_transformed

    else:  # common
        def func_transformed(X):
            return np.array([func(x) for x in X])

        return func_transformed

class SkoBase(metaclass=ABCMeta):
    def register(self, operator_name, operator, *args, **kwargs):
        '''
        regeister udf to the class
        :param operator_name: string
        :param operator: a function, operator itself
        :param args: arg of operator
        :param kwargs: kwargs of operator
        :return:
        '''

        def operator_wapper(*wrapper_args):
            return operator(*(wrapper_args + args), **kwargs)

        setattr(self, operator_name, types.MethodType(operator_wapper, self))
        return self


def swap(individual):
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1], individual[n2] = individual[n2], individual[n1]
    return individual


def reverse(individual):
    '''
    Reverse n1 to n2
    Also called `2-Opt`: removes two random edges, reconnecting them so they cross
    Karan Bhatia, "Genetic Algorithms and the Traveling Salesman Problem", 1994
    https://pdfs.semanticscholar.org/c5dd/3d8e97202f07f2e337a791c3bf81cd0bbb13.pdf
    '''
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1:n2] = individual[n1:n2][::-1]
    return individual


def transpose(individual):
    # randomly generate n1 < n2 < n3. Notice: not equal
    n1, n2, n3 = sorted(np.random.randint(0, individual.shape[0] - 2, 3))
    n2 += 1
    n3 += 2
    slice1, slice2, slice3, slice4 = individual[0:n1], individual[n1:n2], individual[n2:n3 + 1], individual[n3 + 1:]
    individual = np.concatenate([slice1, slice3, slice2, slice4])
    return individual

class PSO_TSP(SkoBase):
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, w=0.8, c1=0.1, c2=0.1):
        self.func = func_transformer(func)
        self.func_raw = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter

        self.w = w
        self.cp = c1
        self.cg = c2

        self.X = self.crt_X()
        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        self.update_pbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.verbose = False

    def crt_X(self):
        tmp = np.random.rand(self.size_pop, self.n_dim)
        return tmp.argsort(axis=1)

    def pso_add(self, c, x1, x2):
        x1, x2 = x1.tolist(), x2.tolist()
        ind1, ind2 = np.random.randint(0, self.n_dim - 1, 2)
        if ind1 >= ind2:
            ind1, ind2 = ind2, ind1 + 1

        part1 = x2[ind1:ind2]
        part2 = [i for i in x1 if i not in part1]  # this is very slow

        return np.array(part1 + part2)

    def update_X(self):
        for i in range(self.size_pop):
            x = self.X[i, :]
            x = self.pso_add(self.cp, x, self.pbest_x[i])
            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

        for i in range(self.size_pop):
            x = self.X[i, :]
            x = self.pso_add(self.cg, x, self.gbest_x)
            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

        for i in range(self.size_pop):
            x = self.X[i, :]
            new_x_strategy = np.random.randint(3)
            if new_x_strategy == 0:
                x = swap(x)
            elif new_x_strategy == 1:
                x = reverse(x)
            elif new_x_strategy == 2:
                x = transpose(x)

            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            # self.update_V()
            self.recorder()
            self.update_X()
            # self.cal_y()
            # self.update_pbest()
            # self.update_gbest()

            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y