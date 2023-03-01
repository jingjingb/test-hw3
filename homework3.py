############################################################
# CIS 521: Homework 3
############################################################

# Include your imports here, if any are used.

student_name = "Jingjing Bai"

############################################################
# Imports
############################################################
import random
import copy
import queue
import math

############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(rows, cols):
  return TilePuzzle([[row*cols + col + 1 if row*cols + col + 1<cols*rows else 0 for col in range(cols)] for row in range(rows)])


class TilePuzzle(object):
    
    def __init__(self, board):
        self.board = board
        self.r = len(board)
        self.c = len(board[0])
        for i in range(self.r):
            for j in range(self.c):
                if board[i][j] == 0:
                    self.loc = (i, j)
        self.sol = self.solved_board()
        self.h = 0
        self.f = 0
        self.g = 0
        self.route = []

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        loc = self.loc
        if direction == "up":
            if loc[0] == 0:
                return False
            else:
                self.board[loc[0]][loc[1]] = self.board[loc[0] - 1][loc[1]]
                self.board[loc[0] - 1][loc[1]] = 0
                self.loc = (loc[0] - 1, loc[1])
                return True
        elif direction == "down":
            if loc[0] == self.r - 1:
                return False
            else:
                self.board[loc[0]][loc[1]] = self.board[loc[0] + 1][loc[1]]
                self.board[loc[0] + 1][loc[1]] = 0
                self.loc = (loc[0] + 1, loc[1])
                return True
        elif direction == "left":
            if loc[1] == 0:
                return False
            else:
                self.board[loc[0]][loc[1]] = self.board[loc[0]][loc[1] - 1]
                self.board[loc[0]][loc[1] - 1] = 0
                self.loc = (loc[0], loc[1] - 1)
                return True
        elif direction == "right":
            if loc[1] == self.c - 1:
                return False
            else:
                self.board[loc[0]][loc[1]] = self.board[loc[0]][loc[1] + 1]
                self.board[loc[0]][loc[1] + 1] = 0
                self.loc = (loc[0], loc[1] + 1)
                return True
        return False

    def scramble(self, num_moves):
        directions = ["up", "down", "left", "right"]
        for i in range(num_moves):
            self.perform_move(random.choice(directions))

    def is_solved(self):
        solved = create_tile_puzzle(self.r, self.c)
        if self.board == solved.get_board():
            return True
        return False

    def copy(self):
        return TilePuzzle(copy.deepcopy(self.board))

    def successors(self):
        p = self.copy()
        if p.perform_move("up"):
            yield ("up", p)
        p = self.copy()
        if p.perform_move("down"):
            yield ("down", p)
        p = self.copy()
        if p.perform_move("left"):
            yield ("left", p)
        p = self.copy()
        if p.perform_move("right"):
            yield ("right", p)

    def solved_board(self):
        board = []
        new = []
        cnt = 1
        for i in range(0, self.r):
            for j in range(0, self.c):
                new.append(cnt)
                cnt += 1
            board.append(new)
            new = []
        board[self.r - 1][self.c - 1] = 0
        return board

    # Required
    def find_solutions_iddfs(self):
        is_found_solution = False
        limit = 0
        while not is_found_solution:
            for move in self.iddfs_helper(limit, []):
                yield move
                is_found_solution = True
            limit += 1

    def iddfs_helper(self, limit, route):
        if self.board == self.sol:
            yield route
        elif len(route) < limit:
            for move, puzzle in self.successors():
                for sol in puzzle.iddfs_helper(limit, route + [move]):
                    yield sol

    # Required
    def find_solution_a_star(self):
        open_set = queue.PriorityQueue()
        # f function = g+h,cost so far=g, route, node
        open_set.put((self.manhattan(self.sol), 0, [], self))
        closed_set = {tuple(tuple(x) for x in self.board): 0}

        while open_set:
            curr = open_set.get()
            if tuple(tuple(x) for x in curr[3].board) in closed_set:
                continue
            else:
                closed_set[tuple(tuple(x) for x in curr[3].board)] = curr[1]

            if curr[3].is_solved():
                return curr[2]

            for move, puzzle in curr.successors():
                new_g = curr[1] + 1
                new_h = puzzle.manhattan(puzzle.sol)
                if tuple(tuple(x) for x in puzzle.board) not in closed_set or new_g < closed_set[tuple(tuple(x) for x in puzzle.board)]:
                    closed_set[tuple(tuple(x) for x in puzzle.board)] = new_g
                    open_set.put((new_g + new_h, new_g, curr[2] + [move], puzzle))

    def manhattan(self, t1):
        total = 0
        pos = {}

        for x in range(self.r):
            for y in range(self.c):
                pos[t1[x][y]] = (x, y)

        for x in range(self.r):
            for y in range(self.c):
                a = self.board[x][y]
                pos2 = pos[a]
                total += abs(x - pos2[0]) + abs(y - pos2[1])
        return total


           

"""
class TilePuzzle(object):
    # Required
    def __init__(self, board):
        self.board = [[board[row][col] for col in range(len(board[0]))] for row in range(len(board))]
        self._TilePuzzle__board = board
        self.find_empty = False
        for row in range(len(board)):
            if self.find_empty:
                break
            for col in range(len(board[0])):
                if board[row][col] == 0:
                    self.empty_row = row
                    self.empty_col = col
                    self.find_empty = True
                    break

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        try:
            delta = {"up": [-1, 0],"down": [1,0],"left":[0,-1],"right":[0,1]}[direction]
            delta_row, delta_col = delta[0], delta[1]
            target_row = self.empty_row + delta_row
            target_col = self.empty_col + delta_col
        except:
            return False
        if target_row >=0 and target_row <len(self.board) and target_col>=0 and target_col<len(self.board[0]):
            self.board[self.empty_row][self.empty_col] = self.board[target_row][target_col]
            self.board[target_row][target_col] = 0
            self.empty_col = target_col
            self.empty_row = target_row
            return True
        else:
            return False

    def scramble(self, num_moves):
        for _ in range(num_moves):
            self.perform_move(random.choice(["up", "down", "left", "right"]))

    def is_solved(self):
        return self.board == create_tile_puzzle(len(self.board), len(self.board[0])).get_board()

    def copy(self):
        new_board = copy.deepcopy(self.__board)
        return TilePuzzle(new_board)

    def successors(self):
        for move in ['up','down','left','right']:
            new_p = self.copy()
            if new_p.perform_move(move):
                yield move,new_p

    # Required
    def find_solutions_iddfs(self):
        is_found_solution = False
        limit = 0
        while not is_found_solution:
            for move in self.iddfs_helper(limit, []):
                yield move
                is_found_solution = True
            limit += 1

    def iddfs_helper(self, limit, route):
        if self.is_solved():
            yield route
        elif len(route) < limit:
            for move, puzzle in self.successors():
                for sol in puzzle.iddfs_helper(limit, route + [move]):
                    yield sol


    def heuristic_md(self):
      # manhattan distance
      solved_row = lambda row, col: int((self.board[row][col] - 1)/len(self.board)) \
                    if self.board[row][col] != 0 else len(self.board) - 1
      solved_col = lambda row, col: self.board[row][col] - solved_row(row, col) * len(self.board[0]) - 1 \
                    if self.board[row][col] != 0 else len(self.board[0]) - 1
      return sum([ abs(solved_col(row, col) - col) + abs(solved_row(row, col) - row) \
              for col in range(len(self.board[0])) \
              for row in range(len(self.board)) \
      ])

    # Required
    def find_solution_a_star(self):
        pq = queue.PriorityQueue()
        pq.put((self.heuristic_md(), 0, [], self))
        trace = set()
        while True:
            node = pq.get()
            if tuple(tuple(x) for x in node[3].board) in trace:
                continue
            else:
                trace.add(tuple(tuple(x) for x in node[3].board))
            if node[3].is_solved(): 
                return node[2]
            for (move, new_p) in node[3].successors():
                if tuple(tuple(x) for x in new_p.board) not in trace:
                    pq.put((node[1] + 1 + new_p.heuristic_md(), node[1] + 1, node[2] + [move], new_p))
"""
############################################################
# Section 2: Grid Navigation
############################################################
class GridNavigation(object):

    def __init__(self, start, goal, scene):
        self.pos = start
        self.goal = goal
        self.scene = scene
        self.row_num = len(scene)
        self.col_num = len(scene[0])

    def perform_move(self, direction):
        if direction == "up" and self.pos[0] > 0 \
           and self.scene[self.pos[0] - 1][self.pos[1]] is False:
            self.pos = (self.pos[0] - 1, self.pos[1])
            return True

        if direction == "down" and self.pos[0] < self.row_num - 1 \
           and self.scene[self.pos[0] + 1][self.pos[1]] is False:
            self.pos = (self.pos[0] + 1, self.pos[1])
            return True

        if direction == "left" and self.pos[1] > 0 \
           and self.scene[self.pos[0]][self.pos[1] - 1] is False:
            self.pos = (self.pos[0], self.pos[1] - 1)
            return True

        if direction == "right" and self.pos[1] < self.col_num - 1 \
           and self.scene[self.pos[0]][self.pos[1] + 1] is False:
            self.pos = (self.pos[0], self.pos[1] + 1)
            return True

        if direction == "up-left" and self.pos[0] > 0 and self.pos[1] > 0 \
           and self.scene[self.pos[0] - 1][self.pos[1] - 1] is False:
            self.pos = (self.pos[0] - 1, self.pos[1] - 1)
            return True

        if direction == "up-right" and self.pos[0] > 0 and self.pos[1] < self.col_num - 1 \
           and self.scene[self.pos[0] - 1][self.pos[1] + 1] is False:
            self.pos = (self.pos[0] - 1, self.pos[1] + 1)
            return True

        if direction == "down-left" and self.pos[0] < self.row_num - 1 and self.pos[1] > 0 \
           and self.scene[self.pos[0] + 1][self.pos[1] - 1] is False:
            self.pos = (self.pos[0] + 1, self.pos[1] - 1)
            return True

        if direction == "down-right" and self.pos[0] < self.row_num - 1 and self.pos[1] < self.col_num - 1 \
           and self.scene[self.pos[0] + 1][self.pos[1] + 1] is False:
            self.pos = (self.pos[0] + 1, self.pos[1] + 1)
            return True        

        return False

    def is_solved(self):
        return self.pos == self.goal

    def copy(self):
        return GridNavigation(copy.deepcopy(self.pos), self.goal, self.scene)

    def successors(self):
        for direction in ["up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right"]:
            p = self.copy()
            if p.perform_move(direction):
                yield (direction, p.pos, p)

    def heuristic_ed(self):
        return math.sqrt((self.pos[0] - self.goal[0]) ** 2 + (self.pos[1] - self.goal[1]) ** 2)

    def find_path_a_star(self):
        if self.scene[self.pos[0]][self.pos[1]]:    # start at obstacle
            return None
        pq = queue.PriorityQueue()
        pq.put((self.heuristic_ed(), 0, [self.pos], self))      # add the initial state to queue
        trace = set()               # keep trace of pos history
        while True:
            if pq.empty():          # no optimal solution
                return None
            node = pq.get()         # expand according to priority
            if node[3].pos in trace:    # discard the node if having been expanded
                continue
            else:
                trace.add(node[3].pos)  # add node.pos to trace only when firstly expanded
            if node[3].is_solved(): # optimal solution found
                return node[2]
            for (direction, new_pos, new_p) in node[3].successors():
                if new_pos not in trace:    # don't add to the queue if the node has been expanded
                    if direction in ["up", "down", "left", "right"]:    # step cost = 1
                        pq.put((node[1] + 1 + new_p.heuristic_ed(), node[1] + 1, node[2] + [new_pos], new_p))
                    else:           # step cost = sqrt(2)
                        pq.put((node[1] + math.sqrt(2) + new_p.heuristic_ed(), \
                                node[1] + math.sqrt(2), node[2] + [new_pos], new_p))

def find_path(start, goal, scene):
    p = GridNavigation(start, goal, scene)
    return p.find_path_a_star()


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

class solve_disks(object):
    
    def __init__(self, length, n, grid):    # usually grid = 0 or [] when initialzing
        self.length = length
        self.n = n
        if grid == 0 or len(grid) == 0:
            self.grid =  [i for i in range(n)] + [-1] * (length - n)
        elif len(grid) == length:
            self.grid = grid
        else:
            return

    def perform_move(self, i, steps):
        if (i + steps) < 0 or (i + steps) >= self.length:
            return
        self.grid[i + steps] = self.grid[i]     # steps = -2, -1, 1, 2
        self.grid[i] = -1

    def is_solved(self):
        for i in range(self.length - self.n):
            if self.grid[i] != -1:
                return False
        for i in range(self.length - self.n, self.length):
            if self.grid[i] != self.length - i - 1:
                return False
        return True

    def copy(self):
        return solve_disks(self.length, self.n, copy.deepcopy(self.grid))

    def successors(self):
        for i in range(self.length):
            if (self.grid[i] != -1 and (i + 1) < self.length \
                and self.grid[i + 1] == -1):
                d = self.copy()
                d.perform_move(i, 1)
                yield (i, i + 1), d
                
            if (self.grid[i] != -1 and (i + 2) < self.length \
                and self.grid[i + 1] != -1 and self.grid[i + 2] == -1):
                d = self.copy()
                d.perform_move(i, 2)
                yield (i, i + 2), d
                
            if (self.grid[i] != -1 and (i - 1) >= 0 \
                and self.grid[i - 1] == -1):
                d = self.copy()
                d.perform_move(i, -1)
                yield (i, i - 1), d
                
            if (self.grid[i] != -1 and (i - 2) >= 0 \
                and self.grid[i - 1] != -1 and self.grid[i - 2] == -1):
                d = self.copy()
                d.perform_move(i, -2)
                yield (i, i - 2), d

    def heuristic(self):
        heuristic = 0
        for i in range(self.length):
            if self.grid[i] != -1:
                heuristic += abs(self.length - 1 - self.grid[i] - i)
        return heuristic

    def find_solution_a_star(self):
        pq = queue.PriorityQueue()
        pq.put((self.heuristic(), 0, [], self))  # add the initial state to queue
        trace = set()               # keep trace of pos history
        while True:
            if pq.empty():          # no optimal solution
                return None
            node = pq.get()         # expand according to priority
            if tuple(node[3].grid) in trace:    # discard the node if having been expanded
                continue
            else:
                trace.add(tuple(node[3].grid))  # add node to trace only when expanded
            if node[3].is_solved(): # optimal solution found
                return node[2]
            for (move, new_p) in node[3].successors():
                if tuple(new_p.grid) not in trace:
                    pq.put((node[1] + 1 + new_p.heuristic(), node[1] + 1, node[2] + [move], new_p))

def solve_distinct_disks(length, n):
    p = solve_disks(length, n, 0)
    return p.find_solution_a_star()



############################################################
# Section 4: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = 24

feedback_question_2 = """
The implementation of A* search.
"""

feedback_question_3 = """
3 projects are too much for me in one week, two would be a proper workload.
"""
