import numpy as np

def get_lowest_weight_node(distance_array:np.ndarray):
    sum_arr = np.array([np.sum(arr) for arr in distance_array])
    node = np.argmin(sum_arr)
    return node

def try1(distance_array:np.ndarray, node:int, cities:int):
    graph_rep = np.zeros(shape=(cities, cities),dtype=np.int16)
    graph_rep[node] = np.ones(cities, np.int16)
    graph_rep[node][node] = 0
    pass

if __name__ == "__main__":
    print(np.eye(4, 1))
