def sumCubes(givenList):
    sum_of_cubes = 0
    for i in range(len(givenList)):
        sum_of_cubes += givenList[i]**3
    
    return sum_of_cubes