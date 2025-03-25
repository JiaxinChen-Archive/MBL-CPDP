import math
import numpy as np

""" NSGAII """


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


# Function to carry out NSGAII's fast non-dominated sort: Looking for the maximum values
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) \
                    or (values1[p] <= values1[q] and values2[p] < values2[q]) \
                    or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) \
                    or (values1[q] <= values1[p] and values2[q] < values2[p]) \
                    or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


def crowding_distance_value(values1, values2, nondominated_solution):
    crowding_distance_values = []
    for i in range(0, len(nondominated_solution)):
        crowding_distance_values.append(
            crowding_distance(values1, values2, nondominated_solution[i][:]))

    return crowding_distance_values


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):  # front is index of the same front
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:]) # Sort the value from small to large

    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    SelectF1 = [values1[i] for i in sorted1]
    SelectF2 = [values2[i] for i in sorted1]
    for k in range(1, len(front)-1):
        Max_F1 = max([values1[i] for i in sorted1])
        Max_F2 = max([values2[i] for i in sorted1])
        Min_F1 = min([values1[i] for i in sorted1])
        Min_F2 = min([values2[i] for i in sorted1])

        if Max_F1 - Min_F1 == 0 or Max_F2 - Min_F2 == 0:
            distance[k] = 0
        else:
            distance[k] = distance[k] + (SelectF1[k + 1] - SelectF1[k - 1]) / (Max_F1 - Min_F1) + (
                        SelectF2[k - 1] - SelectF2[k + 1]) / (Max_F2 - Min_F2)

    distance1 = [0 for i in range(0, len(front))]
    for i in range(0, len(front)):
        for j in range(0, len(sorted1)):
            if front[i] == sorted1[j]:
                distance1[i] = distance[j]
                break
    return distance1


def crossover(self, a1, b1, seed):
    proC, disC = 0.9, 15
    parent1, parent2 = self.theta_set[a1], self.theta_set[b1]
    max_theta, min_theta = self.max_theta, self.min_theta
    dimension = np.size(parent1)
    beta = np.zeros(dimension)

    rng = np.random.RandomState(seed)
    mu = rng.rand(dimension)

    for i in range(dimension):
        if mu[i] <= 0.5:
            beta[i] = (2*mu[i])**(1/(disC+1))
        else:
            beta[i] = (2-2*mu[i])**(-1/(disC+1))

    beta = beta*(-1)**rng.randint(0, 2, (1, dimension))
    beta = beta.reshape(-1)
    beta[rng.rand(dimension) < 0.5] = 1
    beta[rng.rand(dimension) > proC] = 1
    solution = (parent1+parent2)/2 + beta*(parent1-parent2)/2

    return mutation(solution, mu, max_theta, min_theta, rng)


def mutation(solution, mu, max_theta, min_theta, rng):
    proM, disM = 1, 20
    D = np.size(solution)
    Site = rng.rand(D) < proM/D
    temp = Site & (mu <= 0.5)
    solution = solution.reshape(-1)
    solution[solution < min_theta] = min_theta
    solution[solution > max_theta] = max_theta
    solution[temp] = solution[temp] + (max_theta - min_theta)*((2*mu[temp]\
                    + (1-2*mu[temp])*(1-(solution[temp]-min_theta)/(max_theta\
                    - min_theta))**(disM+1))**(1/(disM+1))-1)
    temp = Site & (mu > 0.5)
    solution[temp] = solution[temp] + (max_theta - min_theta)*(1-(2*(1-mu[temp])+2*(mu[temp]-0.5)\
                    *(1-(max_theta-solution[temp])/(max_theta-min_theta))**(disM+1))**(1/(disM+1)))

    return solution



def offspringselect(Fv1, Fv2, parentlen, FixedNum):

    nondominated_solution = fast_non_dominated_sort(Fv1, Fv2)
    crowding_distance_values = crowding_distance_value(Fv1, Fv2, nondominated_solution)

    new_solution = []
    for i in range(0, len(nondominated_solution)):
        nondominated_solution_1 = [index_of(nondominated_solution[i][j], nondominated_solution[i]) \
                                   for j in range(0, len(nondominated_solution[i]))]
        front2 = sort_by_values(nondominated_solution_1[:], crowding_distance_values[i][:])
        front = [nondominated_solution[i][front2[j]] for j in range(0, len(nondominated_solution[i]))]
        front.reverse()

        if FixedNum == 'True':
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == parentlen):
                    break
            if (len(new_solution) == parentlen):
                break
        else:
            for value in nondominated_solution[0]:
                new_solution.append(value)
            break


    return new_solution, nondominated_solution

