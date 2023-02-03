from numpy import random
from dataclasses import dataclass
from sys import argv
import numpy as np


def getFitness(distance_array: list[list[float]], chromosome: list[int]) -> float:
    cost: float = 0
    cost += distance_array[chromosome[0]][len(chromosome)-1]
    for i in range(0, len(chromosome)-1):
        cost += distance_array[chromosome[i]][chromosome[i+1]]
    return cost


def roulette_parents(distance_array: list[list[float]], number_of_parents: int, population_array: list[list[int]], parents: list[list[int]]) -> list[list[int]]:
    chromosome_fitness_array: list[float] = []
    for chromosome in population_array:
        fitness: float = getFitness(distance_array, chromosome)
        chromosome_fitness_array.append(fitness)
    total_fitness: float = 0
    for chromosome_fitness_value in chromosome_fitness_array:
        total_fitness += chromosome_fitness_value
    chromosome_fitness_array_probablity: list[float] = []
    for chromosome_fitness_value in chromosome_fitness_array:
        chromosome_fitness_array_probablity.append(
            chromosome_fitness_value/total_fitness)
    chromosome_fitness_array_probablity_partial_sum: list[float] = []
    chromosome_fitness_array_probablity_partial_sum.append(
        chromosome_fitness_array_probablity[0])
    for i in range(1, len(chromosome_fitness_array_probablity)):
        chromosome_fitness_array_probablity_partial_sum.append(
            chromosome_fitness_array_probablity_partial_sum[i-1]+chromosome_fitness_array_probablity[i])
    for i in range(0, number_of_parents):
        random_value: float = random.rand()
        counter: int = 0
        while (chromosome_fitness_array_probablity_partial_sum[counter] < random_value):
            counter += 1
        parents.append(population_array[counter])
    return parents


def tournament_selection(distance_array: list[list[float]], number_of_parents: int, population_array: list[list[int]], parents: list[list[int]]) -> list[list[int]]:
    temporary_population: list[list[int]] = []
    if (len(population_array) <= number_of_parents):
        for chromosomes in population_array:
            parents.append(chromosomes)
        return parents
    else:
        lenght_of_population: int = len(population_array)
        for i in range(0, int(lenght_of_population/2)):
            first_element_fitness: float = getFitness(
                distance_array, population_array[i])
            second_element_fitness: float = getFitness(
                distance_array, population_array[lenght_of_population-i-1])
            if (first_element_fitness < second_element_fitness):
                temporary_population.append(population_array[i])
            else:
                temporary_population.append(
                    population_array[lenght_of_population-i-1])
    return tournament_selection(distance_array, number_of_parents,
                                temporary_population, parents)


def eletism(distance_array: list[list[float]], number_of_parents: int, population_array: list[list[int]], parents: list[list[int]]) -> list[list[int]]:
    for i in range(0, number_of_parents):
        parents.append(population_array[i])
    for i in range(number_of_parents, len(population_array)):
        for j in range(0, len(parents)):
            if (getFitness(distance_array, parents[j]) > getFitness(distance_array, population_array[i])):
                parents[j] = population_array[i]
                break
    for i in range(0, number_of_parents):
        for j in range(0, len(parents)):
            if (getFitness(distance_array, parents[j]) > getFitness(distance_array, population_array[i])):
                parents[j] = population_array[i]
                break
    return parents


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


@dataclass
class GeneticAlgorithm():

    cities: int = 100

    def partial_crossover(self, tour1, tour2) -> tuple[list[int], list[int]]:
        size: int = self.cities
        crossover_point_1: int = size//3
        crossover_point_2: int = 2 * (size//3)

        # MAKING CHILD_1
        repMap1: dict[int, int] = {}
        for i in range(crossover_point_1, crossover_point_2+1):
            repMap1[tour1[i]] = tour2[i]
        child_1: list[int] = tour2[0: crossover_point_1] + \
            tour1[crossover_point_1:crossover_point_2+1] + \
            tour2[crossover_point_2+1:]
        for i in range(0, crossover_point_1):
            if (child_1[i] in repMap1.keys()):
                while (child_1[i] in repMap1.keys()):
                    child_1[i] = repMap1[child_1[i]]
        for i in range(crossover_point_2+1, self.cities):
            if (child_1[i] in repMap1.keys()):
                while (child_1[i] in repMap1.keys()):
                    child_1[i] = repMap1[child_1[i]]

        # MAKING CHILD 2
        repMap2: dict[int, int] = {}
        for i in range(crossover_point_1, crossover_point_2+1):
            repMap2[tour2[i]] = tour1[i]
        child_2: list[int] = tour1[0: crossover_point_1] + \
            tour2[crossover_point_1:crossover_point_2+1] + \
            tour1[crossover_point_2+1:]
        for i in range(0, crossover_point_1):
            if (child_2[i] in repMap2.keys()):
                while (child_2[i] in repMap2.keys()):
                    child_2[i] = repMap2[child_2[i]]
        for i in range(crossover_point_2+1, self.cities):
            if (child_2[i] in repMap2.keys()):
                while (child_2[i] in repMap2.keys()):
                    child_2[i] = repMap2[child_2[i]]
        return (child_1, child_2)


if __name__ == "__main__":
    input_dataset = argv[1]

    vals: tuple[str, int, list[tuple[float, float]], list[list[float]]
                ] = fileReader(input_dataset)

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
    eletism(vals[3], 30, population, fittest_elements)
    print(getFitness(vals[3], [34, 92, 81, 21, 39, 24, 47, 5, 76, 6, 95, 69, 83, 68, 31, 49, 23, 14, 91, 51, 65, 0, 38, 19, 4, 57, 85, 10, 18, 13, 52, 86, 7, 56, 82, 3, 94, 40, 67, 89, 35, 78, 42, 90, 44, 45, 53,
          80, 62, 71, 58, 37, 43, 87, 79, 70, 50, 61, 77, 59, 17, 99, 33, 84, 15, 64, 55, 9, 30, 2, 93, 12, 66, 32, 46, 8, 73, 75, 88, 20, 25, 36, 29, 98, 41, 48, 63, 1, 54, 97, 22, 16, 26, 72, 11, 28, 60, 27, 74, 96]))
    population.pop()
    population.append([34, 92, 81, 21, 39, 24, 47, 5, 76, 6, 95, 69, 83, 68, 31, 49, 23, 14, 91, 51, 65, 0, 38, 19, 4, 57, 85, 10, 18, 13, 52, 86, 7, 56, 82, 3, 94, 40, 67, 89, 35, 78, 42, 90, 44, 45, 53, 80,
                      62, 71, 58, 37, 43, 87, 79, 70, 50, 61, 77, 59, 17, 99, 33, 84, 15, 64, 55, 9, 30, 2, 93, 12, 66, 32, 46, 8, 73, 75, 88, 20, 25, 36, 29, 98, 41, 48, 63, 1, 54, 97, 22, 16, 26, 72, 11, 28, 60, 27, 74, 96])

    best_fitness: float = 10**8
    fittest_element = []
    xyz = 0
    while best_fitness > 1600:
        prevfitness = best_fitness
        xyz = xyz+1
        for i in range(0, len(fittest_elements), 2):
            algs: GeneticAlgorithm = GeneticAlgorithm(cities=vals[1])
            crossover_ele: tuple[list[int], list[int]] = algs.partial_crossover(
                tour1=fittest_elements[i], tour2=fittest_elements[i+1])
            fittest_elements.append(crossover_ele[0])
            fittest_elements.append(crossover_ele[1])
        copy_fittest_elements: list[list[int]] = []

        eletism(vals[3], 30,
                fittest_elements, copy_fittest_elements)
        fittest_elements = []
        for i in copy_fittest_elements:
            fittest_elements.append(i)
            if (getFitness(vals[3], i) < best_fitness):
                best_fitness = getFitness(vals[3], i)
                fittest_element = i
        if (xyz % 25 == 0):
            for i in fittest_elements:
                copy_ele1: list[int] = i
                random_num_1 = np.random.randint(0, vals[1])
                random_num_2 = np.random.randint(0, vals[1])
                temp = copy_ele1[random_num_1]
                copy_ele1[random_num_1] = copy_ele1[random_num_2]
                copy_ele1[random_num_2] = temp
                if (getFitness(vals[3], copy_ele1) < getFitness(vals[3], i)):
                    i = copy_ele1
        if (prevfitness != best_fitness):
            print(best_fitness)
            prevfitness = best_fitness
    print(best_fitness)
    print(fittest_element)
