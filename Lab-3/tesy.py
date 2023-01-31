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
vals = fileReader("D: \AI LAB\AI-Lab\Lab-3\euc_100")

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


for i in range(0, 10, 2):
    algs: GeneticAlgorithm = GeneticAlgorithm(cities=vals[1])
    crossover_ele = algs.partial_crossover(
        tour1=fittest_elements[i], tour2=fittest_elements[i+1])
    for i in crossover_ele:
        print(i)
        print()
