def getEdge(graph: list[list[int]], index: int) -> int:
    for i in range(0, len(graph[index])):
        if (graph[index][i] == 1):
            return i
    return -1


def getTour(graph: list[list[int]], tour: list[int], index: int) -> None:
    if (getEdge(graph, index) == tour[0]):
        return
    else:
        tour.append(index)
        getTour(graph, tour, getEdge(graph, index))
