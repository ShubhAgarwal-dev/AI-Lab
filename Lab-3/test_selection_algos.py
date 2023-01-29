from selection_algos import *

distance_array: list[list[float]] = [[0, 3.9, 4.0, 6.1, 9.8], [3.9, 0, 5.2, 8.7, 2.1], [
    4.0, 5.2, 0, 6.4, 3.1], [6.1, 8.7, 6.4, 0, 8.8], [9.8, 2.1, 3.1, 8.8, 0]]
number_of_parents: int = 4
population_array: list[list[int]] = [[0, 1, 2, 3, 4],
                                     [1, 0, 2, 3, 4],
                                     [2, 0, 1, 3, 4],
                                     [0, 2, 1, 3, 4],
                                     [1, 2, 0, 3, 4],
                                     [2, 1, 0, 3, 4],
                                     [2, 1, 3, 0, 4],
                                     [1, 2, 3, 0, 4],
                                     [3, 2, 1, 0, 4],
                                     [2, 3, 1, 0, 4],
                                     [1, 3, 2, 0, 4],
                                     [3, 1, 2, 0, 4],
                                     [3, 0, 2, 1, 4],
                                     [0, 3, 2, 1, 4],
                                     [2, 3, 0, 1, 4],
                                     [3, 2, 0, 1, 4],
                                     [0, 2, 3, 1, 4]]

parents: list[list[int]] = []
print(roulette_parents(distance_array,
                       number_of_parents, population_array, parents))

parents: list[list[int]] = []
print(tournament_selection(distance_array,
                           number_of_parents, population_array, parents))

parents: list[list[int]] = []
print(eletism(distance_array,
              number_of_parents, population_array, parents))
