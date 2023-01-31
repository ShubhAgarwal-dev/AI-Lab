def fileReader(path: str) -> tuple[str, int, list[tuple[float, float]], list[list[float]]]:
    file = open("D:\\AI LAB\\AI-Lab\\Lab-3\\euc_100", "r")
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
    return (type_of_distance, number_of_cities, coordinate_array, adjacency_matix)
