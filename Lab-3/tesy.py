import numpy as np
from algos import GeneticAlgorithm
from file_reader import array_converter
from timeit import default_timer
from selection_algos import *
from file_reader_modified import fileReader
from selection_algos import getFitness
import time


# vals = array_converter("D:\AI LAB\AI-Lab\Lab-3\euc_100")

# print(vals[3])
vals: tuple[str, int, list[tuple[float, float]], list[list[float]]
            ] = fileReader("D: \\AI LAB\\AI-Lab\\Lab-3\\euc_100")

population: list[list[int]] = []
while (len(population) < 100):
    testGuess: list[int] = []
    while (len(testGuess) < vals[1]):
        x: int = np.random.randint(0, vals[1])
        if (x not in testGuess):
            testGuess.append(x)
    if (testGuess not in population):
        population.append(testGuess)


fittest_elements: list[list[int]] = []
eletism(vals[3], 10, population, fittest_elements)

# for i in fittest_elements:
#     print(i)
#     print()

best_fitness: float = 10**8

for xyz in range(0, 20000):
    for i in range(0, len(fittest_elements), 2):
        algs: GeneticAlgorithm = GeneticAlgorithm(cities=vals[1])
        crossover_ele: tuple[list[int], list[int]] = algs.partial_crossover(
            tour1=fittest_elements[i], tour2=fittest_elements[i+1])
        fittest_elements[i] = crossover_ele[0]
        fittest_elements[i+1] = crossover_ele[1]

    copy_fittest_elements: list[list[int]] = []

    eletism(vals[3], 10,
            fittest_elements, copy_fittest_elements)
    fittest_elements = []
    for i in copy_fittest_elements:
        fittest_elements.append(i)
        if (getFitness(vals[3], i) < best_fitness):
            best_fitness = getFitness(vals[3], i)
    if (xyz % 25 == 0):
        for i in fittest_elements:
            random_num_1 = np.random.randint(0, 100)
            random_num_2 = np.random.randint(0, 100)
            temp = i[random_num_1]
            i[random_num_1] = i[random_num_2]
            i[random_num_2] = temp

print(best_fitness)


# PARTIAL CROSSOVER TEST CODE

# arr1: list[int] = [1, 2, 5, 6, 4, 3, 8, 7]
# arr2: list[int] = [1, 4, 2, 3, 6, 5, 7, 8]

# algs: GeneticAlgorithm = GeneticAlgorithm(cities=8)
# print(algs.partial_crossover(arr1, arr2))
