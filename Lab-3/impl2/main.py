from stimulatedAnnealing_2 import SA_TSP
import numpy as np
from pathlib import Path
from typing import Union


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

def fileReader(path: str) -> tuple[str, int, list[tuple[float, float]], list[list[float]]]:
    file = open(path, "r")
    type_of_distance: str = file.readline().strip('\n')
    number_of_cities: int = int(file.readline().strip('\n'))
    coordinate_array: list[tuple[float, float]] = []
    # making coordinate array
    for i in range(0, number_of_cities):
        coordinate_split: list[str] = file.readline().strip('\n').split(" ")
        new_coordinate_tuple: tuple[float, float] = (
            float(coordinate_split[0]), float(coordinate_split[1]))
        coordinate_array.append(new_coordinate_tuple)
    # making adjacency matrix
    adjacency_matix: list[list[float]] = []
    for i in range(0, number_of_cities):
        distance_split: list[str] = file.readline().strip('\n').split(" ")
        distance_split_modified: list[float] = []
        for j in range(0, number_of_cities):
            distance_split_modified.append(float(distance_split[j]))
        adjacency_matix.append(distance_split_modified)
    file.close()
    return (type_of_distance, number_of_cities, coordinate_array, adjacency_matix)

_, num_points, car, distance_matrix = array_converter(r"D:\Projects\AI Lab\Lab-3\dataset\euc_100")


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=1000, T_min=1, L=100 * num_points)

best_points, best_distance = sa_tsp.run()
print(best_points, best_distance, cal_total_distance(best_points))

