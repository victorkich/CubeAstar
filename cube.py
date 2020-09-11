from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
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


class Node:
    """
        Basic informations about each node
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
        Class where the A* will be run inside and plotting
        in real time the trajectory reached.
    """

    def __init__(self, aniplot=True):
        threading.Thread.__init__(self)
        self.ax = plt.gca(projection='3d')
        self.graph_list = []
        self.aniplot = aniplot
        self.start_point = (None, None, None)
        self.end_point = (None, None, None)

    def run(self):
        for _ in tqdm(range(LOOP)):
            factible = False
            while not factible:
                start_point, end_point, generated_grid = self.generate_environment()
                factible, total_time, astar_path = self.astar(generated_grid,  start_point, end_point)
                time.sleep(0.03)

            graph_dict = {'k': len(astar_path), 'time': total_time}
            self.graph_list.append(graph_dict)

        self.aniplot = False
        time.sleep(.5)


    def astar(self, grid, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        time_start = time.time()

        # Create start and end node
        start_node = Node(None, start)
        end_node = Node(None, end)
        self.st_node = start_node
        self.ed_node = end_node

        # Initialize both open and closed list
        self.open_list = []
        self.closed_list = []

        # Add the start node
        self.open_list.append(start_node)

        # Loop until you find the end
        while len(self.open_list) > 0:
            if self.aniplot:
                time.sleep(0.015)
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

            if len(self.open_list) > GRID**2:
                return [False, [], []]

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                time_end = time.time() - time_start
                return True, time_end, path[::-1]  # Return total time and reversed path

            # Lista de possiveis movimentos
            movements = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)]
            # Generate children
            children = []
            for new_position in movements:  # Adjacent squares

                # Get node position
                node_position = tuple(current_node.position[i] + new_position[i] for i in range(3))
                # Make sure within range
                if any([True for i in range(3) if node_position[i] > GRID-1 or node_position[i] < 0]):
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
                child.h = sum([(child.position[i] - end_node.position[i]) ** 2 for i in range(3)])
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in self.open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                self.open_list.append(child)

    def generate_environment(self):
        p_grid = np.random.choice(a=[0, 1], size=(GRID, GRID, GRID), p=[1.-GAMMA, GAMMA])

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

    def animate(self, i):
        if self.aniplot:
            self.ax.clear()
            p_o = [o.position for o in self.open_list]
            p_c = [c.position for c in self.closed_list]
            df_o = DataFrame(p_o, columns={'x', 'y', 'z'})
            df_c = DataFrame(p_c, columns={'x', 'y', 'z'})
            self.ax.scatter3D(df_o['x'], df_o['y'], df_o['z'], color='blue', linewidths=1)
            self.ax.scatter3D(df_c['x'], df_c['y'], df_c['z'], color='cyan', linewidths=2)
            s_p = self.st_node.position
            e_p = self.ed_node.position
            self.ax.scatter3D(s_p[0], s_p[1], s_p[2], color='green', linewidths=5)
            self.ax.scatter3D(e_p[0], e_p[1], e_p[2], color='red', linewidths=5)

            title = 'Realtime Trajectory:'
            self.ax.set_title(title)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
            self.ax.set_xlim([0, GRID])
            self.ax.set_ylim([0, GRID])
            self.ax.set_zlim([0, GRID])
            time.sleep(0.01)
        else:
            plt.close()
            graph = DataFrame(self.graph_list)
            graph = graph.groupby('k').mean()
            plt.plot(graph.index, graph['time'])
            plt.title('Distance(un) x Time(s) on {} environments with {}% of obstacles:'.format(LOOP, GAMMA * 100))
            plt.legend(['Time'])
            plt.show()


def main():
    alg = Algorithm(aniplot=PLOT)
    alg.setDaemon(True)
    alg.start()
    time.sleep(0.5)
    if PLOT:
        ani = FuncAnimation(fig, alg.animate, interval=1)
        plt.show()


if __name__ == '__main__':
    main()
