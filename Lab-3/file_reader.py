import numpy as np
from pathlib import Path
from typing import Union

def array_converter(file_loc:Union[str, Path]):
    with open(file=file_loc, mode='r') as file:
        file_content = file.readlines()
        dis_type = file_content[0].strip()
        num_args = int(file_content[1].strip())
        coordinates_array = np.zeros(shape=(num_args, 2), dtype=np.float128)
        for i in range(num_args):
            coordinates_array[i][0], coordinates_array[i][1] = (file_content[i+2].strip()).split(" ")
        distance_array = np.zeros(shape=(num_args, num_args), dtype=np.float64)
        for i in range(num_args):
            dis = (file_content[i+num_args+2].strip()).strip(" ")
            for j in range(num_args):
                distance_array[i][j] = dis[j]
        return (dis_type, num_args, coordinates_array, distance_array)
        