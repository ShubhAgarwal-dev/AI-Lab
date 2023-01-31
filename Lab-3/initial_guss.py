import numpy as np
from queue import Queue


def get_lowest_weight_node(distance_array:np.ndarray):
    sum_arr = np.array([np.sum(arr) for arr in distance_array])
    node = np.argmin(sum_arr)
    return int(node)


def cost_calc(distance_array:np.ndarray, tour):
    dist = 0
    prev_node = tour[0]
    for ind, node in enumerate(tour):
        if ind!=0:
            dist += distance_array[prev_node][node]
    return int(dist)


def parent(connection, node:int) ->  int:
    if connection[node] != node:
        return parent(connection, connection[node])
    return node


def tour(graph, initial_city:int):
    # this function needs some improvement
    tour_arr = [initial_city]
    print(graph[initial_city])
    node = int(np.argmax(graph[initial_city]))
    graph[initial_city][node] = 0
    graph[node][initial_city] = 0
    tour_arr.append(node)
    while np.sum(graph[node]) != 0:
        old_node = node
        node = int(np.argmax(graph[node]))
        tour_arr.append(node)
        graph[old_node][node] = 0
        graph[node][old_node] = 0
        print(graph[node])
    return tour_arr


def try1(distance_array:np.ndarray, node:int, cities:int):
    graph_rep = np.zeros(shape=(cities, cities),dtype=np.int16)
    # for i in range(cities):
    #     graph_rep[i][node] = 1
    # graph_rep[node] = np.ones(cities, np.int16)
    # graph_rep[node][node] = 0
    # dis = distance_array[node].copy()
    connection = np.zeros(cities, np.int16)
    for i in range(cities):
        connection[i] = i
    savings = np.zeros(((cities*cities-cities)//2, 3), np.float64)
    indexer = 0
    for i in range(cities):
        for j in range(cities):
            if i>j:
                savings[indexer] = [i, j, distance_array[i][node]+distance_array[j][node]-distance_array[i][j]]
                indexer += 1
    savings = savings[savings[:, 2].argsort()][::-1]
    for saving in savings:
        i, j, _ = saving
        i = int(i)
        j = int(j)
        if (parent(connection, i) != parent(connection, j)) and _ > 0:
            graph_rep[i][j] = 1
            graph_rep[j][i] = 1
            connection[parent(connection, j)] = i
    return tour(graph_rep, node)


def get_edges(distance_array:np.ndarray, cities:int):
    edges = np.zeros(shape=((cities**2-cities)//2, 3), dtype=np.float64)
    index=0
    for i in range(cities):
        for j in range(cities):
            if i>j:
                edges[index]=[distance_array[i][j], i, j]
                index+=1
    return edges


def greedy(distance_array:np.ndarray, cities:int):
    edges = get_edges(distance_array, cities)
    graph = np.zeros((cities, cities), np.int8)
    edges.sort(axis=0)
    print(edges)
    connection = np.zeros(cities, bool)
    for edge in edges:
        edge_1 = int(edge[1])
        edge_2 = int(edge[2])
        if not connection[edge_1] or not connection[edge_2]:
            if np.sum(connection) == cities:
                break
            connection[edge_1] = True
            connection[edge_2] = True
            graph[edge_1][edge_2] = 1
            graph[edge_2][edge_2] = 1
    print(graph)
    return get_bfs_tour(graph, get_lowest_weight_node(distance_array))


def get_bfs_tour(graph, initial_city:int):
    to_visit_queue = Queue(maxsize=0)
    to_visit_queue.put(initial_city)
    checker = []
    checker.append(initial_city)
    visited = []
    while not to_visit_queue.empty():
        to_visit = to_visit_queue.get()
        checker.remove(to_visit)
        visited.append(to_visit)
        neighbours = [ind for ind, val in enumerate(graph[visited]) if int(val)==1]
        for neighbour in neighbours:
            if neighbour not in checker and neighbour not in visited:
                to_visit_queue.put(neighbour)
                checker.append(neighbour)
    
    return visited



def farthest_insertion():
    """complete this algorithm"""
    pass

if __name__ == "__main__":
    from file_reader import array_converter

    d, n, c, dis_arr = array_converter(r"Lab-3\euc_100")
    # tourr = try1(dis_arr, get_lowest_weight_node(dis_arr), n)
    tourr = greedy(dis_arr, n)
    print(tourr)
    print(cost_calc(dis_arr, tourr))
