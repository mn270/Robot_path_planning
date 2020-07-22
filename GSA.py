"""
Example how to use a Gravitational Search Algorithm to path planning
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Parameters
N = 30  # number of agents
G0 = 20  # G0 gravity 50
G = G0
Eps = 0.001  # small constant ???
Xs = -45  # START position
Ys = -45
Xg = 45  # GOAL position
Yg = 45
alfa = 1000  # penalty
Area = 100  # size area
robot_radius = 4.0
Episodes = 600.0  # simulation time
zet = 0.1  # zeta parameter exp to change gravity
SHOW = 3000
len = Area / 2
xmax = ymax = len
xmin = ymin = -len


class obtacle():
    """
    Create static obstacles
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class agent():
    """
    Create agent (one planet, object, mass)
    """
    def __init__(self, x, y):
        self.x = []
        self.y = []
        self.fit_mat = []
        self.m = 0.0
        self.M = 0.0
        self.x.append(Xs)
        self.y.append(Ys)
        self.x.append(Xs + random.randint(-3, 3))
        self.y.append(Ys + random.randint(-3, 3))
        self.fit = math.sqrt(pow(Xg - Xs, 2) + pow(Yg - Ys, 2))
        self.Vx = random.randint(-3, 3)
        self.Vy = random.randint(-3, 3)
        self.ax = 0.0
        self.ay = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.Succes = 0
        self.colision = False

    def calc_position(self, t):
        """
        Calculate actual position 
        :param t:
        :return:
        """
        x = math.ceil(self.x[t] + self.Vx)
        y = math.ceil(self.y[t] + self.Vy)
        x += random.randint(-1, 1)
        y += random.randint(-1, 1)
        if (self.fit < 5):
            self.y.append(Yg)
            self.x.append(Xg)
            self.Succes = 1

        else:
            self.y.append(y)
            self.x.append(x)

    def calc_fit(self, obtacles, t):
        colision = False
        if ((self.x[t] - self.x[t - 1]) == 0):
            A = 1
            B = self.x[t]
            C = 0
        else:
            A = (self.y[t] - self.y[t - 1]) / (self.x[t] - self.x[t - 1])
            C = self.y[t - 1] - self.x[t - 1] * (self.y[t] - self.y[t - 1]) / (self.x[t] - self.x[t - 1])
            B = -1
        d = math.sqrt(pow(A, 2) + pow(B, 2))
        if (self.x[t - 1] > self.x[t]):
            Xup = self.x[t - 1]
            Xdown = self.x[t]
        else:
            Xup = self.x[t]
            Xdown = self.x[t - 1]
        if (self.y[t - 1] > self.y[t]):
            Yup = self.y[t - 1]
            Ydown = self.x[t]
        else:
            Yup = self.y[t]
            Ydown = self.y[t - 1]
        for i, _ in enumerate(obtacles):
            if ((abs(A * obtacles[i].x + B * obtacles[i].y + C) / d < robot_radius / 2) and (
                    obtacles[i].x < Xup + robot_radius / 2) and (obtacles[i].x > Xdown - robot_radius / 2) and (
                    obtacles[i].y < Yup + robot_radius / 2) and (obtacles[i].y > Ydown - robot_radius / 2)):
                colision = True
        if (colision):
            self.fit = math.sqrt(pow(Xg - self.x[t], 2) + pow(Yg - self.y[t], 2)) + alfa
            self.colision = True
        else:
            self.fit = math.sqrt(pow(Xg - self.x[t], 2) + pow(Yg - self.y[t], 2))
        self.fit_mat.append(self.fit)

    def calc_m(self, worst, bw):
        k = self.fit - worst
        if (k == 0):
            k = -random.random()
        self.m = k / bw
        # self.m = k / N

    def calc_M(self, sum_m):
        self.M = self.m / sum_m

    def calc_V(self):
        self.Vx = random.random() * self.Vx + self.ax
        self.Vy = random.random() * self.Vy + self.ay

    def calc_a(self):
        self.ax = self.Fx / self.M
        self.ay = self.Fy / self.M

    def calc_F(self, agents, i, t, limit_k):
        self.Fx = 0
        self.Fy = 0
        for n in range(N):
            if n != i:
                if (agents[n].fit < limit_k):
                    dx = agents[n].x[t] - self.x[t]
                    dy = agents[n].y[t] - self.y[t]
                    E = math.sqrt(pow(dx, 2) + pow(dy, 2))
                    GME = G * self.M * agents[n].M / (E + Eps)
                    Fx = GME * dx
                    self.Fx = random.random() * Fx + self.Fx
                    Fy = GME * dy
                    self.Fy = random.random() * Fy + self.Fy


def path_planning(agents, obtacles, G):
    SUCCES = 0
    plt.plot(Xs, Ys, "*k")
    plt.plot(Xg, Yg, "*m")
    t = 0
    while ((t < Episodes) and (N / 2 > SUCCES)):
        t += 1
        SUCCES = 0
        worst = agents[0].fit
        best = agents[0].fit
        b = 0
        for i in range(N):
            agents[i].calc_fit(obtacles, t)
            if (agents[i].fit > worst):
                worst = agents[i].fit
            if (agents[i].fit < best):
                best = agents[i].fit
                b = i
        SUM_m = 0
        bw = best - worst
        for i in range(N):
            agents[i].calc_m(worst, bw)
            SUM_m += agents[i].m

        for i in range(N):
            agents[i].calc_M(SUM_m)
            agents[i].calc_F(agents, i, t, worst + 3 * bw / 4)
            agents[i].calc_a()
            agents[i].calc_V()

        for i in range(N):
            agents[i].calc_position(t)
            SUCCES += agents[i].Succes
            if (t > SHOW):
                plt.plot(agents[i].x[t - 1], agents[i].y[t - 1], ".w")
                plt.plot(agents[i].x[t], agents[i].y[t], ".b")
                plt.pause(0.1)

        G = G0 * math.exp(zet * -t / Episodes)

    D = []
    index = []
    fit_data = []
    L = 0
    for i in range(N):
        d = 0
        if (not agents[i].colision):
            for n in range(t - 1):
                d += math.sqrt(
                    pow(agents[i].x[n + 1] - agents[i].x[n], 2) + pow(agents[i].y[n + 1] - agents[i].y[n], 2))
            D.append(math.ceil(d))
            index.append(i)
            L += 1
    id = D.index(min(D))
    id = index[id]

    for n in range(t):
        s = 0
        for i in range(L):
            s += agents[index[i]].fit_mat[n]
        fit_data.append(s / L)

    return id, t, fit_data, min(D)


def main():
    print("potential_field_planning start")
    obtacles = []
    agents = []
    viel = random.randint(20, 30)
    plt.grid(True)
    plt.axis("equal")

    for i in range(viel):
        obtacles.append(obtacle(random.randint(-len, len), random.randint(-len, len)))
        plt.plot(obtacles[i].x, obtacles[i].y, "o")

    plt.grid(True)
    plt.axis("equal")
    for _ in range(N):
        agents.append(agent(Xs, Ys))

    id, t, data, path = path_planning(agents, obtacles, G)

    plt.cla()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    k = 9
    for i in range(viel):
        ax1.plot(obtacles[i].x, obtacles[i].y, "o")
    ax1.plot(Xs, Ys, "*k")
    ax1.plot(Xg, Yg, "*m")

    ax1.plot(agents[id].x, agents[id].y, linewidth=1)
    ax1.set_title('path')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(agents[id].fit_mat, color='tab:blue', label='best_fit')
    ax.plot(data, color='tab:orange', label='average_fit')
    ax.set_title('fit function')
    fig.legend(loc='up right')

    print(t)
    print(path)

    while (1):
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()

    print(__file__ + " Done!!")
