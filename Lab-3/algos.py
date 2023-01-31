from dataclasses import dataclass
import numpy as np


def get_legal_label(num, child1, child2, part: int, part2: int):
    index = -1
    for i in range(part, part2):
        if num == child1[i]:
            index = i
            break
    opp_num = child2[index]
    if opp_num in child1:
        return get_legal_label(opp_num, child1, child2, part, part2)
    else:
        return opp_num


@dataclass
class GeneticAlgorithm():

    cities: int = 100

    # def partial_crossover(self, tour1, tour2):
    #     child1 = np.zeros(shape=self.cities, dtype=np.int16)
    #     child2 = np.zeros(shape=self.cities, dtype=np.int16)
    #     partition = self.cities // 3
    #     child2[partition:self.cities -
    #            partition] = tour1[partition:self.cities-partition]
    #     child1[partition:self.cities -
    #            partition] = tour2[partition:self.cities-partition]
    #     child1[:partition] = tour1[:partition]
    #     child1[self.cities-partition:] = tour1[self.cities-partition:]
    #     child2[:partition] = tour2[:partition]
    #     child2[self.cities-partition:] = tour2[self.cities-partition:]
    #     for ind, val in enumerate(child1):
    #         if ind > partition and ind < self.cities-partition:
    #             continue
    #         if np.sum(np.where(child1 == val, 1, 0)) >= 2:
    #             child1[ind] = get_legal_label(
    #                 val, child1, child2, partition, self.cities-partition)

    #     for ind, val in enumerate(child2):
    #         if ind > partition and ind < self.cities-partition:
    #             continue
    #         if np.sum(np.where(child2 == val, 1, 0)) >= 2:
    #             child2[ind] = get_legal_label(
    #                 val, child2, child1, partition, self.cities-partition)
    #     return (child1, child2)

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
