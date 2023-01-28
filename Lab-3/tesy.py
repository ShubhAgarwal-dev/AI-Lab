from timeit import default_timer

initial_clock = default_timer()

from  file_reader import array_converter
vals = array_converter(r".\Lab-3\euc_100")

from algos import GeneticAlgorithm
import numpy as np

parent1 = np.array([2, 5, 6, 3, 1, 7, 8, 4, 9], dtype=np.int16)
parent2 = np.array([3, 4, 1, 7, 9, 2, 8, 5, 6], dtype=np.int16)

gen = GeneticAlgorithm(cities=9)
print(gen.partial_crossover(parent1, parent2))
end_clock = default_timer()
print(end_clock - initial_clock)
