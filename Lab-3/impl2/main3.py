from stimulatedAnnealing_2 import SA_TSP
from PSO_TSP import PSO_TSP
import numpy as np
from pathlib import Path
from typing import Union
from timeit import default_timer

start = default_timer()

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


_, num_points, points_coordinate, distance_matrix = array_converter(r"D:\Projects\AI Lab\Lab-3\dataset\euc_100")


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

pso_tsp = PSO_TSP(func=cal_total_distance, n_dim=num_points, size_pop=200, max_iter=800, w=0.1, c1=0.7, c2=0.7)
best_points, best_distance = pso_tsp.run()

print(best_points, best_distance)


end = default_timer()
print(f"Time:{end-start}")

sa_tsp = SA_TSP(func=cal_total_distance, x0=best_points, T_max=1000, T_min=1, L=10*num_points)

best_points, best_distance = sa_tsp.run()
print(best_points, best_distance, cal_total_distance(best_points))


end = default_timer()
print(f"Time:{end-start}")


