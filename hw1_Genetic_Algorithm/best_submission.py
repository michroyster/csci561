from math import sqrt
import random, time

# Read and format input
def get_input():
    file = open('input.txt', 'r')
    contents = file.readlines()
    file.close()
    num = int(contents[0].strip())
    cities = contents[1:]
    for i, c in enumerate(cities):
        cities[i] = list(map(int, c.split()))
    return (num, cities)

# Because i'm lazy
def get_size():
    file = open('input.txt', 'r')
    contents = file.readlines()
    file.close()
    return int(contents[0].strip())

# Write solution to file
def write_solution(solution, cities):
    file = open('output.txt', 'w')
    for s in solution:
        file.write(f"{cities[s][0]} {cities[s][1]} {cities[s][2]}\n")
    file.write(f"{cities[solution[0]][0]} {cities[solution[0]][1]} {cities[solution[0]][2]}\n")
    file.close()

# Calculate distance between two cities
def distance(City1, City2):
    return sqrt((City2[0]-City1[0])**2 + (City2[1]-City1[1])**2 + (City2[2]-City1[2])**2)

# Calculate total distance of a path
def total_distance(path, cities):
    dist = 0
    for i in range(len(path)-1):
        dist += distance(cities[path[i]], cities[path[i+1]])
    dist += distance(cities[path[0]], cities[path[-1]])
    return dist

# verify that the path has each city only once
def verify(path):
    temp = set()
    for p in path:
        temp.add(p)
    return len(temp) == len(path)

# Evaluate fitness of the given path
def fitness(path, cities):
    dist = 0
    for i in range(len(path)-1):
        dist += distance(cities[path[i]], cities[path[i+1]])
    dist += distance(cities[path[0]], cities[path[-1]])
    return 1000000/dist

# Find an MST and transform into path
def prim(adjacency_list, size, initial):
    infinity = float('inf')
    # adjacency_list = adjacency(cities, size)
    visited = [0 for n in range(size)]
    visited[initial] = 1
    mst = []
    edge_count = 0
    while (edge_count < size - 1):
        shortest = infinity
        start, end = 0, 0
        for i in range(size):
            if visited[i]:
                for j in range(size):
                    if (not visited[j] and adjacency_list[i][j]):
                        if shortest > adjacency_list[i][j]:
                            shortest = adjacency_list[i][j]
                            start = i
                            end = j
        mst.append((start,end))
        visited[end] = 1
        edge_count += 1
    return mst_to_path(mst)

# Create Adjacency list for Prim's
def adjacency(cities, size):
    adjList = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            adjList[i][j] = distance(cities[i], cities[j])
    return adjList

# Convert MST to valid path
def mst_to_path(mst):
    path = []
    path.append(mst[0][0])
    for m in mst:
        path.append(m[1])
    return path

# Create initial population of cities and ranked list
def CreateInitialPopulation(pop_size, cities):
    size = len(cities)
    initial_population = []
    ranked_list = []
    # initial_guesses = 10
    # count = size // initial_guesses
    count = 5
    adjacency_list = adjacency(cities, size)
    for i in range(count):
        temp = prim(adjacency_list, size, int(size*(i*(1/count))))
        initial_population.append(temp)
        ranked_list.append((fitness(temp,cities),i))

    for i in range (count, pop_size):
        temp = RandomPath(size)
        initial_population.append(temp)
        ranked_list.append((fitness(temp,cities),i))
    ranked_list = sorted(ranked_list, reverse=True)
    return initial_population, ranked_list

# Generate a random path from initial city to each city and returning to initial city
def RandomPath(size):
    positions = [x for x in range(size)]
    path = []
    for i in range(size):
        selection = random.choice(positions)
        path.append(selection)
        positions.remove(selection)
    return path

# Create a ranking list based on the given population
def CreateRankedList(population, cities):
    ranked_list = []
    for i, p in enumerate(population):
        ranked_list.append((fitness(p,cities),i))
    ranked_list = sorted(ranked_list, reverse=True)
    return ranked_list

# Creates a mating pool by Roulette wheel selection. size can be controlled
def CreateMatingPool(population, rankList, apex_pool, random_pool) -> list:
    size = len(population)
    mating_pool = []
    
    for i in range(apex_pool):
        mating_pool.append(population[rankList[i][1]])
    
    total = 0
    for r in rankList:
        total += r[0]
    ticker = random.uniform(0.0, total)
    prob = 0
    for i in range(len(population) - (apex_pool*2) - random_pool):
        for j, r in enumerate(rankList):
            prob += r[0]
            if (prob >= ticker):
                mating_pool.append(population[r[1]])
                # rankList.pop(j) # make sure not to pick the same twice 
                # total -= r[0]
                prob = 0
                ticker = random.uniform(0.0, total)
                break
    random.shuffle(mating_pool)
    return mating_pool

# Breed two paths together
def Crossover(Parent1, Parent2):
    size = len(Parent1)
    a = random.randint(0,size)
    b = random.randint(0,size)
    city_start = min(a,b)
    city_end = max(a,b)
    # print(f"{city_start}-{city_end}")
    
    parent1_gene = Parent1[city_start:city_end]
    parent2_gene = [gene for gene in Parent2 if gene not in parent1_gene]

    child=[]
    for i in range(city_start):
        child.append(parent2_gene.pop(0))
    child += parent1_gene
    for i in range(size-len(parent1_gene)-city_start):
        child.append(parent2_gene.pop(0))

    return child

# Swap two cities
def Mutate(child):
    size = len(child)
    a, b = random.randint(0, size-1), random.randint(0,size-1)
    child[a], child[b] = child[b], child[a]

# The main course Genetic Algorithm
def GeneticAlgorithm(generation_size, apex_pool, random_pool, max_generations, mutate_chance):
    path_size, cities = get_input()
    population, ranked_list = CreateInitialPopulation(generation_size, cities)
    
    g = 0
    best_fit = 0
    while(g < max_generations):
        new_population = []
        mating_pool = CreateMatingPool(population, ranked_list, apex_pool, random_pool)
        
        for i in range(apex_pool):
            new_population.append(population[ranked_list[i][1]])

        n = len(mating_pool)
        mid = n//2
        positions = [x for x in range(n)]
        for i in range(n):
            parent1 = mating_pool[i]
            parent2 = mating_pool[(i+mid)%n]
            child1 = Crossover(parent1, parent2)
            if(mutate_chance > random.random()*100):
                Mutate(child1)
            new_population.append(child1)
        
        new_pop_size = len(new_population)
        while (new_pop_size < generation_size):
            new_population.append(RandomPath(path_size))
            new_pop_size +=1
        population = new_population
        ranked_list = CreateRankedList(population,cities)
        g += 1
        if g % (10000//path_size) == 0:
            if best_fit == ranked_list[0][0]:
                print(f"CONVERGENCE at {g}")
                break
            else:
                best_fit = ranked_list[0][0]
        # print(f"Generation {g}: {total_distance(population[ranked_list[0][1]], cities)}\t| population size: {len(population)}")
    # for i,r in enumerate(ranked_list):
    #     print(f"{1000000*(r[0]**-1)} {r[1]}")
    #     if i > 20:
    #         break
    write_solution(population[ranked_list[0][1]], cities)
    return (population[ranked_list[0][1]], fitness(population[ranked_list[0][1]],cities))

def main():
    begin1 = time.time() # Only for measuring performance
    size, cities = get_input()
    gen_size = size
    if gen_size > 400: gen_size = 400
    apex_pool=int(gen_size*0.05)
    if apex_pool < 1 : apex_pool = 1
    random_pool=int(gen_size*0.025)
    if random_pool < 1 : random_pool = 1
    mutate_chance = 0.025
    max_generations = 10000
    solution, solution_fitness = GeneticAlgorithm(generation_size=gen_size, apex_pool=apex_pool, random_pool=random_pool, max_generations=max_generations, mutate_chance=mutate_chance)
    # logging performance
    # file = open("performance_log.txt", 'a')
    # file.write(f"ExeTime: {time.time() - begin1} | NumCities: {size} | GenSize: {gen_size} | ApexPool: {apex_pool} | RandomPool: {random_pool} | MutateChance: {mutate_chance} | MaxGenerations {max_generations} | Distance: {total_distance(solution, cities)}\n") 
    # file.close()

if __name__ == '__main__':
    begin = time.time()
    main()
    print(f"Execution time: {time.time() - begin}")