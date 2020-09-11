from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
import numpy as np
import threading
import time

# Configurable Parameters
GRID = 100    # Cube axes length
GAMMA = 0.25  # Obstacles percentage
LOOP = 50     # Successful environments to run
PLOT = True   # Real time plotting

fig = plt.figure()
graph_list = []


class Node:
    """
        Basic information about each node.

            Args:
                parent (Node): this variable carry a father Node object.
                position (tuple): tuple containing x, y and z values.

            Returns:
                bool: return True if this node is in the same position.

    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class Algorithm(threading.Thread):
    """
        Class where the A* will be run inside and plotting in real time the trajectory reached.

            Args:
                aniplot (bool): aniplot is a bool value that enable or disable realtime 3d plotting.

    """

    def __init__(self, aniplot=True):
        threading.Thread.__init__(self)
        self.ax = plt.gca(projection='3d')
        self.graph_list = []
        self.aniplot = aniplot
        self.start_point = (None, None, None)
        self.end_point = (None, None, None)
        self.final_plot = False

    def run(self):
        """
            This is the core loop what is executed in the thread called by .start() wrapper.
            Here the path of all environment is processed.

        """

        for _ in tqdm(range(LOOP)):
            feasible = False
            while not feasible:
                start_point, end_point, generated_grid = generate_environment()
                feasible, total_time, astar_path = self.astar(generated_grid, start_point, end_point)

            graph_dict = {'k': len(astar_path), 'time': total_time}
            self.graph_list.append(graph_dict)

        plt.close()
        global graph_list
        graph_list = self.graph_list

    def astar(self, grid, start, end):
        """
             This is the core function of astar algorithm. Here the path between start and end nodes
             inside the grid is calculated interactively.

                Args:
                    grid (numpy.array): the grid is a binary environment, one array with
                                        shape=(GRID, GRID, GRID) where all of the values either 0 or 1.
                                        1 is the obstacle and 0 is the free space.
                    start (tuple): tuple with position x, y and z
                    end (tuple): tuple with position x, y and z

                Returns:
                    bool: first value on return is the feasible bool. That indicates if the environment is
                    feasible or not.
                    float: second value on return is the time float. This indicates how long it took to
                    process the data and solved the environment.
                    list: third value on return is the path list. This indicates what nodes the path pass to
                    travel from start to end.
        """

        time_start = time.time()

        # Create start and end node
        self.start_node = Node(None, start)
        self.end_node = Node(None, end)

        # Initialize both open and closed list
        self.open_list = []
        self.closed_list = []

        # Add the start node
        self.open_list.append(self.start_node)

        # Loop until you find the end
        while len(self.open_list) > 0:
            if self.aniplot:
                time.sleep(0.05)
            # Get the current node
            current_node = self.open_list[0]
            current_index = 0
            for index, item in enumerate(self.open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            self.open_list.pop(current_index)
            self.closed_list.append(current_node)

            if len(self.open_list) > GRID ** 2:
                return [False, [], []]

            # Found the goal
            if current_node == self.end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                time_end = time.time() - time_start
                return True, time_end, path[::-1]  # Return factible bool, total time and reversed path

            # listing all of possible movements and generate a children list
            movements = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)]
            children = []
            for new_position in movements:  # Adjacent squares
                # Get node position
                node_position = tuple(current_node.position[i] + new_position[i] for i in range(3))
                # Make sure within range
                if any([True for i in range(3) if node_position[i] > GRID - 1 or node_position[i] < 0]):
                    continue

                # Make sure walkable terrain
                if grid[node_position] != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)
                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                for closed_child in self.closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = sum([(child.position[i] - self.end_node.position[i]) ** 2 for i in range(3)])
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in self.open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                self.open_list.append(child)

    def animate(self, i):
        """
            Plot in realtime a 3d graph with scatters using matplotlib. This only occur if self.aniplot
            had True.

                Args:
                    self (object): just a self object to get the others variables of the class

        """

        if self.aniplot:
            try:
                self.ax.clear()
                o_l = self.open_list
                c_l = self.closed_list
                df_o = DataFrame([o.position for o in o_l])
                df_c = DataFrame([c.position for c in c_l])
                self.ax.scatter3D(xs=df_o[0], ys=df_o[1], zs=df_o[2], color='blue', linewidths=1)
                self.ax.scatter3D(xs=df_c[0], ys=df_c[1], zs=df_c[2], color='cyan', linewidths=2)
                s_p = self.start_node.position
                e_p = self.end_node.position
                self.ax.scatter3D(xs=s_p[0], ys=s_p[1], zs=s_p[2], color='green', linewidths=5)
                self.ax.scatter3D(xs=e_p[0], ys=e_p[1], zs=e_p[2], color='red', linewidths=5)

                title = 'Realtime Trajectory:'
                self.ax.set_title(title)
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('y')
                self.ax.set_zlabel('z')
                self.ax.set_xlim([0, GRID])
                self.ax.set_ylim([0, GRID])
                self.ax.set_zlim([0, GRID])
            except:
                pass

        time.sleep(0.015)


def generate_environment():
    """
        This function generate an environment using global parameters how input, because that it don't
        have arguments as input. GRID and GAMMA create one numpy.array carrying binary values with tree
        dimensions. Where 0 is the free positions and 1 is obstacles. This function also generate two
        point: start and end.

            Returns:
                tuple: start point containing x, y and z coordinates.
                tuple: end point containing x, y and z coordinates.
                numpy.array: binary grid of environment with shape=(GRID, GRID, GRID).

    """

    p_grid = np.random.choice(a=[0, 1], size=(GRID, GRID, GRID), p=[1. - GAMMA, GAMMA])

    start_end = [False, False]
    while not all(start_end):
        point = tuple(np.random.randint((0, 0, 0), (GRID, GRID, GRID)))
        if not p_grid[point]:
            if not start_end[0]:
                p_start = point
                start_end[0] = True
            else:
                p_end = point
                start_end[1] = True

    return p_start, p_end, p_grid


def main():
    alg = Algorithm(aniplot=PLOT)
    alg.setDaemon(True)
    alg.start()
    if PLOT:
        _ = FuncAnimation(fig, alg.animate, interval=1)
        plt.show()
    else:
        alg.join()

    graph = DataFrame(graph_list)
    graph = graph.groupby('k').mean()
    plt.plot(graph.index, graph['time'])
    plt.title('Distance(un) x Time(s) on {} environments with {}% of obstacles:'.format(LOOP, GAMMA * 100))
    plt.legend(['Time'])
    plt.show()


if __name__ == '__main__':
    main()
