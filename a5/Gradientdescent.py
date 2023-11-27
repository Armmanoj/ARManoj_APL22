import numpy as np
import matplotlib.pyplot as plt

"""
Steps-
1. n gradient descent
2. Function for differentiation (1D)
3. f(starting point, function, derivatives=NaN)
4. Plot for all 4 situations.
"""


def regression(x, y):
    Ey = np.mean(y)
    Ex = np.mean(x)
    covxy = np.cov(
        x, y
    )  # returns a 2x2 covariance matrix, whose off-diagonal terms is the covariance
    varx = np.var(x)
    slope = covxy[0][1] / varx
    return {"Ey": Ey, "Ex": Ex, "covxy": covxy, "varx": varx, "slope": slope}


def differentiate_cstep(f, c, h=0.001):
    """Complex step differentiation is used to calculate the derivative with O(h^2) accuracy-
    Args:
        f (single variable function analytic at c):
        c (float): f'(c) has to be calculated
        h (float): step size, default 0.001
    """
    return np.imag(f(complex(c, h))) / h


def gradientdescent_g(x, f, gradient, t=0.001, stop=256):
    """Implements gradient descent where the gradient function is given

    Args:
        x (np.array(float)): starting value, as a numpy array
        f (function): function: R^N -> R
        t (np.array(float)): step size, default 0.001
        stop (int): The number of iterations before terminating, default 256
        gradient (function of n inputs): returns gradient of f
    """
    points = (
        []
    )  # all intermediate points in gradient descent are returned for plotting purpose

    for i in range(stop):
        gradf = gradient(*tuple(x))
        x = np.subtract(x, t * gradf)
        points.append(x)
    print(points[-1])
    return points


def gradientdescent_n(x, f, t=0.001, stop=256):
    """Implements gradient descent where the gradient function is numerically evaluated, and f is single variable

    Args:
        x (double): starting value
        f (function): function: R^N -> R
        t (double): step size, default 0.001
        stop (int): The number of iterations before terminating, default 256
    """
    points = (
        []
    )  # all intermediate points in gradient descent are returned for plotting purpose
    gradient = lambda y, h: differentiate_cstep(f, y, h)
    h = 0.001
    gradf = gradient(x, h)
    for i in range(stop):
        x = np.subtract(x, t * gradf)
        h = min(
            0.001, t * gradf * 0.2
        )  # The accuracy with which the gradient should be calculated is different at each point
        if h == 0:  # if gradf=0, we have reached the minimum
            return points
        gradf = gradient(x, h)
        points.append(x)
    print(points[-1])
    return points


def descent_steps_plot_3d(f, startingx, step, gradient, xlim, ylim, stop):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plotting the surface
    # https://matplotlib.org/stable/gallery/mplot3d/mixed_subplots.html
    # https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html
    sampling = 1000  # The number of x and y samples taken for plotting
    # Define a grid of x and y values
    x = np.linspace(xlim[0], xlim[1], sampling)
    y = np.linspace(ylim[0], ylim[1], sampling)
    X, Y = np.meshgrid(x, y)
    points = gradientdescent_g(startingx, f, gradient, step, stop)
    ax.plot_surface(X, Y, f(X, Y), cmap="viridis", alpha=0.4)

    # Plotting the gradient descent steps
    cmap = plt.get_cmap(
        "copper"
    )  # The closeness of the algorithm to the minimum is represented using color
    ax.scatter3D(
        [i[0] for i in points],
        [i[1] for i in points],
        [f(i[0], i[1]) for i in points],
        c=np.arange(len(points)),
        cmap=cmap,
        s=5,
    )
    # Plot to find an expression for the convergence rate
    # Below log(f(x:kth step)-f(x:optimum)) is plotted vs k
    plt.show()
    fig = plt.figure()
    axb = fig.add_subplot()
    a, b = np.array([i for i in range(1, len(points))], dtype="float"), np.array(
        [
            np.abs(f(points[i][0], points[i][1]) - f(points[-1][0], points[-1][1]))
            for i in range(len(points) - 1)
        ]
    )
    axb.scatter(a, np.log(b))
    bestfit = regression(a, np.log(b))
    axb.plot(a, (bestfit["Ey"] + (bestfit["slope"]) * (a - bestfit["Ex"])), color="r")
    axb.set_xlabel("Number of Steps")
    axb.set_ylabel("Log(error)")
    axb.set_title("Convergence rate")

    # As the graph looks linear, I tried to find a "c" such that error= O((1/c)^k), where k is the number of steps, by taking ratio of successive errors
    c = []
    for i in range(30, len(b) - 31):
        c.append(b[i] / b[i + 1])
    print(
        "The algorithm converges as:\n",
        "O((1/{})^k)".format(np.mean(np.array(c)).round(4)),
    )
    plt.show()


def descent_steps_plot_2d(f, startingx, step, xlim, stop=100):
    x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000)
    z = f(x)
    points = gradientdescent_n(startingx, f, step, stop)
    fig, ax = plt.subplots(2)

    # Plotting the gradient descent
    ax[0].set_ylabel("y")
    ax[0].set_title("Gradient Descent on y=f(x)")
    cmap = plt.get_cmap(
        "copper"
    )  # The closeness of the algorithm to the minimuum is represented using color
    ax[0].scatter(
        points, f(np.array(points)), c=np.arange(len(points)), cmap=cmap, s=20, alpha=1
    )
    ax[0].plot(x, z, alpha=0.5)

    # Plot to find an expression for the convergence rat
    # Below log(f(x:kth step)-f(x:optimum)) is plotted vs k
    x, y = np.array([i for i in range(1, len(points))], dtype="float"), np.array(
        [np.abs(f(points[i]) - f(points[-1])) for i in range(len(points) - 1)]
    )
    ax[1].scatter(x, np.log(y))
    bestfit = regression(x, np.log(y))
    ax[1].plot(x, (bestfit["Ey"] + (bestfit["slope"]) * (x - bestfit["Ex"])), color="r")
    ax[1].set_xlabel("Number of Steps")
    ax[1].set_ylabel("Log(error)")
    ax[1].set_title("Convergence rate")

    # As the graph looks linear, I tried to find a "c" such that error= O((1/c)^k), where k is the number of steps, by taking ratio of successive errors
    c = np.zeros(len(y) - 1)
    for i in range(len(c)):
        c[i] = y[i] / y[i + 1]
    print("The algorithm converges as:\n", "O((1/{})^k)".format(np.mean(c).round(2)))
    fig.subplots_adjust(hspace=0.4)
    plt.show()


# Problem 1


def f1(x):
    return x**2 + 3 * x + 8


xlim1 = [-5, 5]
descent_steps_plot_2d(f1, -3, 0.3, xlim1, 8)

# Problem 4


def f5(x):
    return np.cos(x) ** 4 - np.sin(x) ** 3 - 4 * np.sin(x) ** 2 + np.cos(x) + 1


xlim5 = [0, 2 * np.pi]
descent_steps_plot_2d(f5, 4.3, 0.3, xlim5, 20)

# Problem 2

xlim3 = [-10, 10]
ylim3 = [-10, 10]


def f3(x, y):
    return x**4 - 16 * x**3 + 96 * x**2 - 256 * x + y**2 - 4 * y + 262


def df3_dxi(x, y):
    return np.array([4 * x**3 - 48 * x**2 + 192 * x - 256, 2 * y - 4])


step = np.array([0.05, 0.05])

descent_steps_plot_3d(f3, np.array([1.0, 5.0]), step, df3_dxi, xlim3, ylim3, stop=3000)

# Problem 3
step = np.array([0.05, 0.05])
lim4 = [-np.pi, np.pi]


def f4(x, y):
    return np.exp(-((x - y) ** 2)) * np.sin(y)


def df4_d(x, y):
    return np.array(
        [
            -2 * np.exp(-((x - y) ** 2)) * np.sin(y) * (x - y),
            np.exp(-((x - y) ** 2)) * np.cos(y)
            + 2 * np.exp(-((x - y) ** 2)) * np.sin(y) * (x - y),
        ]
    )


descent_steps_plot_3d(f4, np.array([0.1, -0.1]), step, df4_d, lim4, lim4, stop=175)
