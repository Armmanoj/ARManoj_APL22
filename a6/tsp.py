

import numpy as np
import matplotlib.pyplot as plt
import time

"""
    Step 1: Read the file and store N, list of all cities as a nx2 array.
    Step2: Calculate the distance(cost) function, given a permutation of [1 to n], in efficient way.
    Step3: Define the probability distribution as a function of temperature.
    Step4: Define a cooling with time, or number of iterations sequence.
    Step5: Dynamic programming solution?
    Step6: Define a move function
    Step7: Write algorithm
    Step: Plot the solution
    Step8: Optimize the distance calculation.
"""


def readcities(fname):
    # Reads the txt file and returns
    cities = []
    with open(fname) as f:
        N = int(f.readline())  # Number of cities is read
        city = f.readline().split()
        while city:
            city = np.array([city[0], city[1]], dtype="float")
            cities.append(city)
            city = f.readline().split()
    return np.stack(cities), N


def distance(cities, Order):
    """This calculates the length of a particular path

    Args:
        cities (array): List of coordinates of all cities
        Order (array): Order in which to visit the cities
    """
    citiesnext = np.roll(Order, shift=1, axis=0)
    sqdistances = np.sum((cities[Order] - cities[citiesnext]) ** 2, axis=1)
    return np.sum(np.sqrt(sqdistances))


def prob(T, Delta):
    """Returns probability of a step being in the correct direction._

    Args:
        T (Float): Temperature
        Delta (Float): It is the change in path length due to a move.
    """
    return np.exp(-300 * Delta / T)


def cool(T):
    return 0.9999 * T  # exponential cooling


def MinT(x):
    return 0.0001 * x


def MaxT(x):
    return 10000000 * x


def move(Order, N):
    """This slightly permutes the order of cities, by swapping 2 elements.

    Args:
        N (int): Number of cities
        Order (array): Order of traversing the cities
    """
    Ordernew = np.copy(Order)
    j = np.random.randint(0, N)
    i = np.random.randint(0, N)
    Ordernew[i] = Order[j]
    Ordernew[j] = Order[i]
    return Ordernew


def TSP(fname):
    cities, N = readcities(fname)

    Order = np.random.permutation(N)  # Order of traversing cities
    OrderBest = np.random.permutation(N)  # This holds the best order found
    Ordernew = np.copy(Order)  # This will store the new order of cities

    Dist1 = distance(cities, Order)
    Dinitial = Dist1
    D = [Dist1]  # Holds all the distance in the ith iteration
    Dist2 = 0  # This variable is to store the updated distance
    Dbest = Dist1  # Holds the value of the best distance found

    Tf = MinT(Dist1)  # Final temperature
    T = MaxT(Dist1)  # Initialize a reasonable value of T
    choice = 1  # This is a random variable used to choose whether to accept a new solution or not
    a = 0
    while T > Tf:
        a = a + 1
        Ordernew = move(Order, N)
        Dist2 = distance(cities, Ordernew)
        if Dist2 > Dist1:
            p = prob(T, Dist2 - Dist1)
            choice = np.random.rand()
        else:
            choice = 0
            p = 1
        if choice < p:  # This is automatically true if Dist2<Dist1
            Order = np.copy(Ordernew)
            Dist1 = Dist2
            D.append(Dist1)
        T = cool(T)
        if Dist1 < Dbest:
            Dbest = Dist1
            OrderBest = Ordernew
    print(a)
    return D, OrderBest, Dbest, Dinitial


start_t = time.time()
D, OrderBest, Dbest, Dinitial = TSP("tsp40.txt")
end_t = time.time()
print("The best distance is, ", Dbest)
print("The percent improvement in path length is- ", 100 * (1 - Dbest / Dinitial), "%")

# Plot the distance vs iterations
b = D[0]
c = []
for i in range(len(D)):
    if D[i] < b:
        b = D[i]
    c.append(b)

plt.plot([i for i in range(len(D))], c)
plt.scatter([i for i in range(len(D))], D)
plt.savefig("Convergence.png")
plt.close()
# Plot the optimum path found through all cities
Cities, N = readcities("tsp40.txt")
CitiesBest = np.vstack((Cities[OrderBest], Cities[OrderBest][0, :]))
plt.plot(
    CitiesBest[:, 0],
    CitiesBest[:, 1],
    marker="o",
    markerfacecolor="green",
    color="blue",
)
plt.title("Optimum Path For Salesman")
plt.savefig("Path.png")
print(end_t - start_t)

# The tsp function asked for in the problem


def tsp(fname):
    cities, N = readcities(fname)

    Order = np.random.permutation(N)  # Order of traversing cities
    OrderBest = np.random.permutation(N)  # This holds the best order found
    Ordernew = np.copy(Order)  # This will store the new order of cities

    Dist1 = distance(cities, Order)
    Dist2 = 0  # This variable is to store the updated distance
    Dbest = Dist1  # Holds the value of the best distance found

    Tf = MinT(Dist1)  # Final temperature
    T = MaxT(Dist1)  # Initialize a reasonable value of T
    choice = 1  # This is a random variable used to choose whether to accept a new solution or not
    while T > Tf:
        Ordernew = move(Order, N)
        Dist2 = distance(cities, Order)
        if Dist2 > Dist1:
            p = prob(T, Dist2 - Dist1)
            choice = np.random.rand()
        else:
            choice = 0
            p = 1
        if choice < p:  # This is automatically true if Dist2<Dist1
            Order = np.copy(Ordernew)
            Dist1 = Dist2
        T = cool(T)
        if Dist1 < Dbest:
            Dbest = Dist1
            OrderBest = Ordernew
    return OrderBest
