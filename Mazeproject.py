import pygame
import os
import math
from timeit import default_timer as timer
from queue import PriorityQueue
 
WIDTH = 800
HEIGHT = 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Maze Solver ( Using  Algorithm )")

# Colors
BACKGROUND = (32, 5, 53)
PATH_CLOSE = (87, 10, 87)
PATH_OPEN = (250, 3, 121)
GRID =    (37, 29, 58)
WALL = (20, 6, 254)
START = (255, 255, 255)
END = (0, 0, 0)
PATH = (47, 143, 157)


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = BACKGROUND
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def reset(self):
        self.color = BACKGROUND

    def make_close(self):
        self.color = PATH_CLOSE

    def make_open(self):
        self.color = PATH_OPEN

    def make_wall(self):
        self.color = WALL

    def make_start(self):
        self.color = START

    def make_end(self):
        self.color = END

    def make_path(self):
        self.color = PATH 
    def is_closed(self):
        return self.color == PATH_CLOSE

    def is_opened(self):
        return self.color == PATH_OPEN

    def is_wall(self):
        return self.color == WALL

    def is_start(self):
        return self.color == START

    def is_end(self):
        return self.color == END

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_wall():
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_wall():
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_wall():
            self.neighbours.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():
            self.neighbours.append(grid[self.row][self.col - 1])


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(node_path, current, draw, counter_start):
    pygame.display.set_caption("Maze Solver ( Constructing Path... )")
    path_count = 0

    while current in node_path:
        current = node_path[current]
        current.make_path()
        path_count += 1
        draw()
    counter_end = timer()
    time_elapsed = counter_end - counter_start
    pygame.display.set_caption(
        f'Time Elapsed: {format(time_elapsed, ".2f")}s | Cells Visited: {len(node_path) + 1} | Shortest Path: {path_count + 1} Cells')


def dfs(draw, grid, start, end, counter_start):
    stack = [start]
    visited = set()
    node_path = {}
    while stack:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        current = stack.pop()
        if current == end:
            reconstruct_path(node_path, end, draw, counter_start)
            end.make_end()
            return True

        if current not in visited:
            visited.add(current)
            for neighbor in current.neighbours:
                if neighbor not in visited and neighbor not in stack and not neighbor.is_wall():
                    stack.append(neighbor)
                    node_path[neighbor] = current
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_close()

    pygame.display.set_caption("Maze Solver ( Unable To Find The Target Node ! )")
    return False


def bfs(draw, grid, start, end, counter_start):
    queue = [start]
    visited = set()
    node_path = {}

    while queue:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        current = queue.pop(0)
        if current == end:
            reconstruct_path(node_path, end, draw, counter_start)
            end.make_end()
            return True

        if current not in visited:
            visited.add(current)
            for neighbor in current.neighbours:
                if neighbor not in visited and neighbor not in queue and not neighbor.is_wall():
                    queue.append(neighbor)
                    node_path[neighbor] = current
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_close()

    pygame.display.set_caption("Maze Solver ( Unable To Find The Target Node ! )")
    return False


def astar(draw, grid, start, end, counter_start):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    node_path = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}
    while not open_set.empty():
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(node_path, end, draw, counter_start)
            end.make_end()
            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                node_path[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()

        draw()
        if current != start:
            current.make_close()

    pygame.display.set_caption("Maze Solver ( Unable To Find The Target Node ! )")
    return False


def ucs(draw, grid, start, end, counter_start):
    count = 0
    frontier = PriorityQueue()
    frontier.put((0, count, start))
    node_path = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    frontier_hash = {start}
    while not frontier.empty():
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        current = frontier.get()[2]
        frontier_hash.remove(current)

        if current == end:
            reconstruct_path(node_path, end, draw, counter_start)
            end.make_end()
            return True

        for neighbor in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                node_path[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in frontier_hash:
                    count += 1
                    frontier.put((g_score[neighbor], count, neighbor))
                    frontier_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_close()

    pygame.display.set_caption("Maze Solver ( Unable To Find The Target Node ! )")
    return False


def greedy(draw, grid, start, end, counter_start):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    node_path = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    open_set_hash = {start}
    while not open_set.empty():
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(node_path, end, draw, counter_start)
            end.make_end()
            return True

        for neighbour in current.neighbours:
            temp_g_score = h(neighbour.get_pos(), end.get_pos())

            if temp_g_score < g_score[neighbour]:
                node_path[neighbour] = current
                g_score[neighbour] = temp_g_score
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()

        draw()
        if current != start:
            current.make_close()

    pygame.display.set_caption("Maze Solver ( Unable To Find The Target Node ! )")
    return False

###############

#<======GREEDY=========>#



def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Node(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GRID, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GRID, (i * gap, 0), (i * gap, width))


def draw_grid_wall(rows, grid):
    for i in range(rows):
        for j in range(rows):
            if i == 0 or i == rows - 1 or j == 0 or j == rows - 1:
                spot = grid[i][j]
                spot.make_wall()


def draw(win, grid, rows, width):
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    draw_grid_wall(rows, grid)
    pygame.display.update()


def get_mouse_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def main(win, width):
    ROWS = 30
    grid = make_grid(ROWS, width)
    Start = None
    End = None
    Run = True

    while Run:
        draw(win, grid, ROWS, width)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                Run = False

            if pygame.mouse.get_pressed()[0]:  # [0] left mouse btn
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not Start and spot != End:
                    Start = spot
                    Start.make_start()
                elif not End and spot != Start:
                    End = spot
                    End.make_end()
                elif spot != Start and spot != End:
                    spot.make_wall()

            if pygame.mouse.get_pressed()[2]:  # [2] Right mouse btn
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == Start:
                    Start = None
                if spot == End:
                    End = None

            if e.type == pygame.KEYDOWN:
                if not Start and not End:
                    pygame.display.set_caption("Maze Solver ( Set Start & End Nodes ! )")

                if e.key == pygame.K_c:
                    Start = None
                    End = None
                    pygame.display.set_caption("Maze Solver ")
                    grid = make_grid(ROWS, width)

                if e.key == pygame.K_d and Start and End:
                    counter_start = timer()
                    pygame.display.set_caption("Maze Solver ( Searching DFS... )")
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    dfs(lambda: draw(win, grid, ROWS, width), grid, Start, End, counter_start)

                if e.key == pygame.K_b and Start and End:
                    counter_start = timer()
                    pygame.display.set_caption("Maze Solver ( Searching BFS... )")
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    bfs(lambda: draw(win, grid, ROWS, width), grid, Start, End, counter_start)


                if e.key == pygame.K_u and Start and End:
                    counter_start = timer()
                    pygame.display.set_caption("Maze Solver ( Searching UCS... )")
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    ucs(lambda: draw(win, grid, ROWS, width), grid, Start, End, counter_start)
    

                if e.key == pygame.K_a and Start and End:
                    counter_start = timer()
                    pygame.display.set_caption("Maze Solver ( Searching A*... )")
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    astar(lambda: draw(win, grid, ROWS, width), grid, Start, End, counter_start)

                if e.key == pygame.K_g and Start and End:
                    counter_start = timer()
                    pygame.display.set_caption("Maze Solver ( Searching GREEDY... )")
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    greedy(lambda: draw(win, grid, ROWS, width), grid, Start, End, counter_start)


    pygame.quit()


main(win, WIDTH)
 