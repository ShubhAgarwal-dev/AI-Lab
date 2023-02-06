from PSO_TSP import PSO_TSP
from ACO import ACO_TSP
import numpy as np
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from timeit import default_timer

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

_, num_points, points_coordinate, distance_matrix = array_converter(r"D:\Projects\AI Lab\Lab-3\dataset\noneuc_250")


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

start = default_timer()

# pso_tsp = PSO_TSP(func=cal_total_distance, n_dim=num_points, size_pop=200, max_iter=800, w=0.8, c1=0.1, c2=0.1)
pso_tsp = PSO_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=800, w=0.8, c1=0.1, c2=0.1)


best_points, best_distance = pso_tsp.run()

print(best_points)
print('best_distance', best_distance)
print('best_distance', type(best_distance))

aca = ACO_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=25, max_iter=100,
              distance_matrix=distance_matrix, initial_x=[best_points], initial_y=[best_distance])

best_x, best_y = aca.run()
print(best_x, best_y)

end = default_timer()

print(f"Time:{end-start}")


