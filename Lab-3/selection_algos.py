from numpy import random


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
