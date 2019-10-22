import numpy as np
import pylab as plt

NUMBER_OF_ROWS = 15
NUMBER_OF_COLUMNS = 15
R = np.random.randint(5, size = (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))-6
R[0,0] = -10
R[NUMBER_OF_ROWS-1, NUMBER_OF_COLUMNS-1] = 1
# print("R: ", R)
value_matrix = np.matrix(np.zeros([NUMBER_OF_ROWS,NUMBER_OF_COLUMNS]))
gamma = 1
epsilon = 0.01
merged = False
deltas = []
# actions
# R, L, U, D ~ 0, 1, 2, 3
class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def x(self):
        return self.x
    def y(self):
        return self.y
    def set_x(self, x):
        self.x = x
    def set_y(self, y):
        self.y = y
def goUp(state):
    state.set_y(state.y - 1)
    if (state.y < 0):
        state.set_y(0)
def goDown(state):
    state.set_y( state.y + 1 )
    if (state.y > NUMBER_OF_ROWS-1):
        state.set_y(NUMBER_OF_ROWS-1)
def goRight(state):
    state.set_x( state.x + 1 )
    if (state.x > NUMBER_OF_COLUMNS-1):
        state.set_x( NUMBER_OF_COLUMNS-1 )
def goLeft(state):
    state.set_x( state.x - 1 )
    if (state.x < 0):
        state.set_x( 0 )
def switch_state_by_action(state, action):
    newState = State(state.x, state.y)
    if (action == 0):
        goRight(newState)
    elif (action == 1):
            goLeft( newState )
    elif (action == 2):
        goUp( newState )
    else:
        goDown( newState )
    return newState
def get_reward(state):
    return R[state.y,state.x]
def is_final(state):
    if ((state.x == NUMBER_OF_COLUMNS-1 and state.y == NUMBER_OF_ROWS-1)):
        return True
    return False
def update(value_matrix):
    aux_value_matrix = value_matrix
    delta = 0
    for i in range(NUMBER_OF_COLUMNS):
        for j in range(NUMBER_OF_ROWS):
            current_state = State(i, j)
            if (is_final(current_state)):
                continue
            v = 0
            # go through all actions
            number_actions = 4
            for a in range(4):
                next_state = switch_state_by_action(current_state, a)
                if (next_state.x == current_state.x and next_state.y == current_state.y):
                    number_actions -= 1
                    continue
                v += get_reward(next_state) + gamma * value_matrix[next_state.y, next_state.x]
            delta = max(delta, abs(v/number_actions-aux_value_matrix[current_state.y, current_state.x]))
            aux_value_matrix[current_state.y, current_state.x] = v/number_actions
    if (delta < epsilon):
        global merged
        merged = True
    global deltas
    deltas.append(delta)
    return aux_value_matrix

i = 0
while not merged and i< 10000:
    i += 1
    value_matrix = update(value_matrix)
    if (merged):
        break

print(i)
# print(value_matrix)

plt.plot(deltas)
plt.show()