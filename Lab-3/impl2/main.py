from stimulatedAnnealing_2 import SA_TSP
from ACO import ACA_TSP
import numpy as np
from pathlib import Path
from typing import Union
import pandas as pd
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

# def fileReader(path: str) -> tuple[str, int, list[tuple[float, float]], list[list[float]]]:
#     file = open(path, "r")
#     type_of_distance: str = file.readline().strip('\n')
#     number_of_cities: int = int(file.readline().strip('\n'))
#     coordinate_array: list[tuple[float, float]] = []
#     # making coordinate array
#     for i in range(0, number_of_cities):
#         coordinate_split: list[str] = file.readline().strip('\n').split(" ")
#         new_coordinate_tuple: tuple[float, float] = (
#             float(coordinate_split[0]), float(coordinate_split[1]))
#         coordinate_array.append(new_coordinate_tuple)
#     # making adjacency matrix
#     adjacency_matix: list[list[float]] = []
#     for i in range(0, number_of_cities):
#         distance_split: list[str] = file.readline().strip('\n').split(" ")
#         distance_split_modified: list[float] = []
#         for j in range(0, number_of_cities):
#             distance_split_modified.append(float(distance_split[j]))
#         adjacency_matix.append(distance_split_modified)
#     file.close()
#     return (type_of_distance, number_of_cities, coordinate_array, adjacency_matix)

_, num_points, points_coordinate, distance_matrix = array_converter(r"D:\Projects\AI Lab\Lab-3\dataset\euc_500")


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

# sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)

# best_points, best_distance = sa_tsp.run()
# print(best_points, best_distance, cal_total_distance(best_points))


# from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 2)

# best_points_ = np.concatenate([best_points, [best_points[0]]])
# best_points_coordinate = points_coordinate[best_points_, :]
# ax[0].plot(sa_tsp.best_y_history)
# ax[0].set_xlabel("Iteration")
# ax[0].set_ylabel("Distance")
# ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
#            marker='o', markerfacecolor='b', color='c', linestyle='-')
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1].set_xlabel("Longitude")
# ax[1].set_ylabel("Latitude")
# plt.show()

# # %% Plot the animation
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# best_x_history = sa_tsp.best_x_history

# fig2, ax2 = plt.subplots(1, 1)
# ax2.set_title('title', loc='center')
# line = ax2.plot(points_coordinate[:, 0], points_coordinate[:, 1],
#                 marker='o', markerfacecolor='b', color='c', linestyle='-')
# ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax2.set_xlabel("Longitude")
# ax2.set_ylabel("Latitude")
# plt.ion()
# p = plt.show()


# def update_scatter(frame):
#     ax2.set_title('iter = ' + str(frame))
#     points = best_x_history[frame]
#     points = np.concatenate([points, [points[0]]])
#     point_coordinate = points_coordinate[points, :]
#     plt.setp(line, 'xdata', point_coordinate[:, 0], 'ydata', point_coordinate[:, 1])
#     return line


# ani = FuncAnimation(fig2, update_scatter, blit=True, interval=25, frames=len(best_x_history))
# plt.show()


aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=50, max_iter=150,
              distance_matrix=distance_matrix,
              alpha=5, beta=3, rho=0.5)

best_x, best_y = aca.run()
print(best_x, best_y)

end = default_timer()

print(f"Time:{end-start}")

fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()
