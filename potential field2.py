"""
Example how use algorithm potential field for robotic path planning in dynamic environment.
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters


KP = 2.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
agents = 10


# show_animation = True

def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


class people():
    def __init__(self, sx, sy, gx, gy, reso, rr, str):
        self.ix = sx
        self.iy = sy
        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy
        self.reso = reso
        self.rr = rr
        self.d = np.hypot(self.sx - self.gx, self.sy - self.gy)
        self.str = str

    def calc_potential_field(self, ox, oy, ):
        minx = - AREA_WIDTH
        miny = - AREA_WIDTH
        maxx = AREA_WIDTH
        maxy = AREA_WIDTH
        xw = int(round((maxx - minx) / self.reso))
        yw = int(round(np.divide(np.subtract(maxy, miny), self.reso)))

        # calc each potential
        pmap = [[0.0 for i in range(yw)] for i in range(xw)]

        for ix in range(xw):
            x = np.add(np.multiply(ix, self.reso), minx)

            for iy in range(yw):
                y = np.add(np.multiply(iy, self.reso), miny)
                ug = calc_attractive_potential(x, y, self.gx, self.gy)
                uo = calc_repulsive_potential(x, y, ox, oy, self.rr)
                uf = np.add(ug, uo)
                pmap[ix][iy] = uf

        return pmap, minx, miny

    def potential_field_planning(self, ox, oy, show_animation):

        # calc potential field
        pmap, minx, miny = self.calc_potential_field(ox, oy)

        # search path

        self.ix = round((self.sx - minx) / self.reso)
        self.iy = round((self.sy - miny) / self.reso)
        gix = round((self.gx - minx) / self.reso)
        giy = round((self.gy - miny) / self.reso)

        if show_animation:
            draw_heatmap(pmap)
            plt.plot(self.ix, self.iy, "*k")
            plt.plot(gix, giy, "*m")

        rx, ry = [self.sx], [self.sy]
        self.potential(show_animation, ox, oy)

        return rx, ry

    def potential(self, show_animation, ox, oy):
        motion = get_motion_model()
        pmap, minx, miny = self.calc_potential_field(ox, oy)
        if show_animation:
            draw_heatmap(pmap)
        if self.d >= self.reso:
            minp = float("inf")
            minix, miniy = -1, -1
            for i, _ in enumerate(motion):
                inx = int(self.ix + motion[i][0] + random.random())
                iny = int(self.iy + motion[i][1] + random.random())
                if inx >= len(pmap) or iny >= len(pmap[0]):
                    p = float("inf")  # outside area
                else:
                    p = pmap[inx][iny]
                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny
            self.ix = minix
            self.iy = miniy
            xp = self.ix * self.reso + minx
            yp = self.iy * self.reso + miny
            self.d = np.hypot(self.gx - xp, self.gy - yp)
            self.sx = xp
            self.sy = yp

            #  if show_animation:
            plt.plot(self.ix, self.iy, self.str)
            plt.pause(0.000000001)

        return 0


def main():
    print("potential_field_planning start")

    sx = -28.0  # start x position [m]
    sy = -28.0  # start y positon [m]
    gx = 28.0  # goal x position [m]
    gy = 28.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]
    viel = random.randint(20, 40)
    show = True
    ox = []
    oy = []
    ox_ = []
    oy_ = []
    agent = []
    str = [".g", ".c", ".m", ".y", ".k"]

    robot = people(sx, sy, gx, gy, grid_size, robot_radius, ".r")
    for _ in range(agents):
        agent.append(
            people(random.randint(-26, 26), random.randint(-26, 26), random.randint(-26, 26), random.randint(-26, 26),
                   grid_size, robot_radius, str[random.randint(0, 4)]))
    ox.append(robot.ix)
    oy.append(robot.iy)
    for i in range(agents):
        ox.append(agent[i].ix)
        oy.append(agent[i].iy)

    for _ in range(viel):
        ox.append(random.randint(-26, 26))
        oy.append(random.randint(-26, 26))

    for i in range(len(ox)):
        ox_.append(ox[i])
        oy_.append(oy[i])

    # if show_animation:
    plt.grid(True)
    plt.axis("equal")

    # path generation
    ox_[0] = 0
    oy_[0] = 0
    robot.potential_field_planning(ox_, oy_, show)
    ox[0] = robot.sx
    oy[0] = robot.sy
    ox_[0] = ox[0]
    oy_[0] = oy[0]
    for i in range(agents):
        ox_[i + 1] = 0
        oy_[i + 1] = 0
        agent[i].potential_field_planning(ox_, oy_, False)
        ox[i + 1] = agent[i].sx
        oy[i + 1] = agent[i].sy
        ox_[i + 1] = ox[i + 1]
        oy_[i + 1] = oy[i + 1]
    while robot.d >= robot.reso:
        ox_[0] = 0
        oy_[0] = 0
        ox[0] = robot.sx
        oy[0] = robot.sy
        robot.potential(show, ox_, oy_)
        ox_[0] = ox[0]
        oy_[0] = oy[0]
        for i in range(agents):
            ox[i + 1] = agent[i].sx
            oy[i + 1] = agent[i].sy
            ox_[i + 1] = 0
            oy_[i + 1] = 0
            agent[i].potential(False, ox_, oy_)
            ox_[i + 1] = ox[i + 1]
            oy_[i + 1] = oy[i + 1]
    #  if show_animation:
    print("Goal!!")
    plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
