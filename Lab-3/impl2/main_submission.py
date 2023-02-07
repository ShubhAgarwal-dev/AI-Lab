import numpy as np
from sys import argv
from pathlib import Path
from typing import Union

class ACO:
    def __init__(self, func, n_dim, distance_matrix,
                 size_pop=10, max_iter=20,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_dim  
        self.size_pop = size_pop  
        self.max_iter = max_iter  
        self.alpha = alpha  
        self.beta = beta  
        self.rho = rho

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  

        self.Tau = np.ones((n_dim, n_dim))  
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int16)  
        self.y = None 
        self.generation_best_X, self.generation_best_Y = [], []  
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y  
        self.best_x, self.best_y = None, None

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):  
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  
            for j in range(self.size_pop):  
                self.Table[j, 0] = 0  # star
                for k in range(self.n_dim - 1):  
                    taboo_set = set(self.Table[j, :k + 1])  
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            y = np.array([self.func(i) for i in self.Table])

            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):
                for k in range(self.n_dim - 1):
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                    delta_tau[n1, n2] += 1 / y[j] 
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
                delta_tau[n1, n2] += 1 / y[j] 

            self.Tau = (1 - self.rho) * self.Tau + delta_tau
            if i%10 == 0:
                best_generation = np.array(self.generation_best_Y).argmin()
                self.best_x = self.generation_best_X[best_generation]
                self.best_y = self.generation_best_Y[best_generation]
                print_function(self.best_x)
                print(f"Best Cost Yet: {self.best_y}")

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    fit = run


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def print_arr(arr):
    print(f"{i}, " for i in arr)

def print_function(best_points):
    print(list(best_points))

def array_converter(file_loc: Union[str, Path]):
    with open(file=file_loc, mode='r') as file:
        file_content: list[str] = file.readlines()
        distance_type: str = file_content[0].strip()
        number_of_args: int = int(file_content[1].strip())
        coordinates_array = np.zeros(
            shape=(number_of_args, 2), dtype=np.float64)
        for i in range(number_of_args):
            coordinates_array[i][0], coordinates_array[i][1] = (
                file_content[i+2].strip()).split(" ")
        distance_array = np.zeros(
            shape=(number_of_args, number_of_args), dtype=np.float64)
        for i in range(number_of_args):
            dis: list[str] = (
                file_content[i+number_of_args+2].strip()).split(" ")
            for j in range(number_of_args):
                distance_array[i][j] = dis[j]
        return (distance_type, number_of_args, coordinates_array, distance_array)

if __name__ == "__main__":

    path__ = argv[1]
    ecu, cities, _, distance_matrix = array_converter(path__)

    if ecu == "euclidean":
        aco = ACO(func=cal_total_distance,distance_matrix=distance_matrix, n_dim=cities, size_pop=min(250, cities), max_iter=400, alpha=2, beta=5, rho=0.1)
    else:
        aco = ACO(func=cal_total_distance,distance_matrix=distance_matrix, n_dim=cities, size_pop=min(250, cities), max_iter=300, alpha=5, beta=5, rho=0.05)
        
    best_points,dis = aco.run()

    print_function(best_points)
    print(f"Best Cost:{dis}")


