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
