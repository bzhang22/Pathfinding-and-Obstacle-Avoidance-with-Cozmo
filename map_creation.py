import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import time

class Map:
    def __init__(self, width, height):
        # Define map size
        self.width = width
        self.height = height
        # Create the grid (0 = non-navigable, 1 = street/path, 2 = obstacle)
        self.grid = np.zeros((height, width))
        self.starting_point = None
        self.destination = None

    def add_street(self, x1, y1, x2, y2):
        # Modify the street addition method for dual-lane streets with width 2
        if x1 == x2:  # Vertical dual-lane street
            self.grid[min(y1, y2):max(y1, y2) + 1, x1] = 1
            if x1 + 1 < self.width:  # Ensure the second lane is within the map bounds
                self.grid[min(y1, y2):max(y1, y2) + 1, x1 + 1] = 1
        elif y1 == y2:  # Horizontal dual-lane street
            self.grid[y1, min(x1, x2):max(x1, x2) + 1] = 1
            if y1 + 1 < self.height:  # Ensure the second lane is within the map bounds
                self.grid[y1 + 1, min(x1, x2):max(x1, x2) + 1] = 1

    def add_obstacle(self, x, y):
        # Add an obstacle at a specified location
        self.grid[y, x] = 2

    def delete_obstacle(self, x, y):

        self.grid[y, x] = 1

    def set_current_point(self, x, y):
        # Set the starting point only if it's on a street
        if self.grid[x, y] !=  0:
            self.starting_point = (y, x)
            self.grid[x, y] = 3
        else:
            print("Starting point must be on a street.")

    def set_destination(self, x, y):
        # Set the destination only if it's on a street
        if self.grid[x, y] !=  0:
            self.destination = (y, x)
            self.grid[x, y] = 4
        else:
            print("Destination must be on a street.")

    def display_map(self, path=None):
        # Display the map with an optional path
        plt.figure(figsize=(10, 10))
        cmap = mcolors.ListedColormap(['black', 'white', 'blue', 'lime', 'red'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(self.grid.T, cmap=cmap, norm=norm)  # Transpose the grid for correct (x, y) display

        # Set the tick frequency to 1
        plt.xticks(np.arange(0, self.width, 1))
        plt.yticks(np.arange(0, self.height, 1))

        # Draw the path if provided
        if path:
            for y, x in path:  # Note the order of x and y here
                plt.scatter(x, y, color='black', s=10)  # s is the size of the point

        if self.starting_point:
            y, x = self.starting_point  # Unpacking in the correct order
            plt.scatter(x, y, color='lime', label='Starting Point', edgecolor='black')
        if self.destination:
            y, x = self.destination  # Unpacking in the correct order
            plt.scatter(x, y, color='red', label='Destination', edgecolor='black')

        plt.gca().invert_yaxis()
        plt.grid(which='both')  # Add a grid
        plt.legend()
        plt.show()

    def display_map_dynamic(self, path=None):
        plt.ion()  # Turn on interactive mode
        plt.figure(figsize=(10, 10))
        cmap = mcolors.ListedColormap(['black', 'white', 'blue', 'lime', 'red'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        while True:
            plt.clf()  # Clear the current figure
            plt.imshow(self.grid.T, cmap=cmap, norm=norm)  # Transpose the grid for correct (x, y) display

            # Set the tick frequency to 1
            plt.xticks(np.arange(0, self.width, 1))
            plt.yticks(np.arange(0, self.height, 1))

            # Draw the path if provided
            if path:
                for y, x in path:
                    plt.scatter(x, y, color='black', s=10)

            if self.starting_point:
                y, x = self.starting_point
                plt.scatter(x, y, color='lime', label='Starting Point', edgecolor='black')
            if self.destination:
                y, x = self.destination
                plt.scatter(x, y, color='red', label='Destination', edgecolor='black')

            plt.gca().invert_yaxis()
            plt.grid(which='both')
            plt.legend()
            plt.draw()  # Redraw the current figure
            plt.pause(0.1)
            time.sleep(1)

        plt.ioff()

    def get_adjacent_nodes(self, x, y):
        adjacent_nodes = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and (self.grid[ny, nx] == 1 or self.grid[ny, nx] == 3 or
            self.grid[ny, nx] == 4):
                adjacent_nodes.append((nx, ny))
        #print(f"Adjacent to ({x}, {y}): {adjacent_nodes}")  # Debug print
        return adjacent_nodes

    def heuristic(self, a, b):
        # Manhattan distance on a square grid
        #print(abs(a[0] - b[0]) + abs(a[1] - b[1]))
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def a_star_search(self):
        start = self.starting_point
        goal = self.destination
        #print(goal)

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            #print(f"Current node: {current}")  # Debug print

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_adjacent_nodes(*current):
                tentative_g_score = g_score[current] + 1  # Assuming uniform cost
                #print(f"Checking neighbor: {neighbor}, Tentative G Score: {tentative_g_score}")  # Debug print

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        # Reconstruct the path found by A* Search
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path



