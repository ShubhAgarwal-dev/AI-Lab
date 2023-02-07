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

def mutation_function(individual):
    copy = individual
    n1 = np.random.randint(0,len(copy))
    n2 = np.random.randint(0,len(copy))
    temp = copy[n1]
    copy[n1]=copy[n2]
    copy[n2]=temp
    return copy

if __name__ == "__main__":
    input_dataset = argv[1]

    vals: tuple[str, int, list[tuple[float, float]], list[list[float]]
                ] = fileReader(input_dataset)

    individualArr = [0,4,365,96,472,245,348,36,85,196,323,152,394,489,127,194,216,300
,102,32,111,330,396,329,142,214,339,436,480,374,130,156,333,228,304,361
,2,50,57,120,294,276,90,101,154,308,492,279,312,123,482,407,68,255
,173,467,171,40,197,285,191,408,465,306,53,47,144,141,495,151,217,206
,405,305,286,397,475,449,468,254,222,487,23,450,174,241,257,456,390,76
,94,474,332,426,320,186,368,383,360,455,248,148,377,108,290,124,74,135
,72,116,264,419,238,3,231,166,359,250,48,321,38,233,485,356,181,18
,319,395,221,269,291,246,19,317,43,282,350,479,247,448,140,481,273,281
,45,499,52,458,464,420,95,459,204,145,134,20,430,136,21,362,75,114
,117,98,336,373,65,340,133,442,342,288,309,314,160,61,428,325,244,69
,239,263,49,376,335,446,324,460,328,232,183,132,483,106,172,178,70,137
,203,161,379,315,357,30,89,198,83,190,497,307,176,67,42,84,55,384
,91,425,409,125,201,147,113,46,13,444,34,86,16,242,266,1,205,391
,195,240,268,327,153,31,103,168,10,265,274,347,27,440,343,14,387,400
,54,64,295,490,478,122,79,385,367,24,496,302,275,453,424,163,182,100
,477,62,301,293,358,471,370,92,66,17,371,259,218,435,382,452,427,121
,469,143,56,128,80,220,260,292,126,451,412,498,175,115,334,237,110,253
,71,256,262,283,41,105,341,299,199,229,73,78,457,271,434,234,473,349
,393,8,12,35,267,351,272,165,131,167,311,44,414,219,60,488,215,22
,401,184,461,298,63,326,107,99,417,406,213,261,212,296,338,352,443,211
,463,185,15,249,423,280,189,209,447,104,466,82,109,470,415,441,429,389
,188,433,224,454,289,7,192,208,6,210,438,164,418,11,392,270,381,493
,303,39,422,158,28,187,170,318,26,223,93,112,169,258,366,81,388,177
,398,345,225,284,399,252,149,118,277,445,33,236,235,322,431,179,25,58
,310,491,484,369,313,344,378,439,51,227,5,297,402,375,355,287,476,386
,337,380,354,87,364,226,316,372,278,180,162,138,331,494,9,413,146,202
,157,411,363,346,230,421,416,159,410,88,462,200,207,243,29,432,437,150
,139,59,97,486,251,77,37,404,155,193,119,353,129,403]
    print(getFitness(vals[3],individualArr))
    population: list[list[int]] = []
    population.append(individualArr)
    while (len(population) < 100):
        copyx = individualArr.copy()
        # testGuess: list[int] = mutation_function(copyx)
        # # print(testGuess)
        # # while (len(testGuess) < vals[1]):
        # #     x: int = np.random.randint(0, vals[1])
        # #     if (x not in testGuess):
        # #         testGuess.append(x)
        # if (testGuess not in population):
        #     population.append(testGuess)
        population.append(mutation_function(copyx))
    fittest_elements: list[list[int]] = []
    eletism(vals[3], 30, population, fittest_elements)
    # print(getFitness(vals[3], [34, 92, 81, 21, 39, 24, 47, 5, 76, 6, 95, 69, 83, 68, 31, 49, 23, 14, 91, 51, 65, 0, 38, 19, 4, 57, 85, 10, 18, 13, 52, 86, 7, 56, 82, 3, 94, 40, 67, 89, 35, 78, 42, 90, 44, 45, 53,
    #       80, 62, 71, 58, 37, 43, 87, 79, 70, 50, 61, 77, 59, 17, 99, 33, 84, 15, 64, 55, 9, 30, 2, 93, 12, 66, 32, 46, 8, 73, 75, 88, 20, 25, 36, 29, 98, 41, 48, 63, 1, 54, 97, 22, 16, 26, 72, 11, 28, 60, 27, 74, 96]))
    # fittest_elements.pop()
    # fittest_elements.append([34, 92, 81, 21, 39, 24, 47, 5, 76, 6, 95, 69, 83, 68, 31, 49, 23, 14, 91, 51, 65, 0, 38, 19, 4, 57, 85, 10, 18, 13, 52, 86, 7, 56, 82, 3, 94, 40, 67, 89, 35, 78, 42, 90, 44, 45, 53, 80,
    #                          62, 71, 58, 37, 43, 87, 79, 70, 50, 61, 77, 59, 17, 99, 33, 84, 15, 64, 55, 9, 30, 2, 93, 12, 66, 32, 46, 8, 73, 75, 88, 20, 25, 36, 29, 98, 41, 48, 63, 1, 54, 97, 22, 16, 26, 72, 11, 28, 60, 27, 74, 96])

    best_fitness: float = 10**8
    fittest_element = []
    xyz = 0
    while xyz < 5000:
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
