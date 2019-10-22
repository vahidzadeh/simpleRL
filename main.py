import numpy as np
import pylab as plt

MATRIX_SIZE = 3
NUMBER_OF_STATES = 3

R = np.matrix([
    [30., 30., -10.],
    [15., -25., 30.],
    [-10., 10., 20.]
])

Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
gamma = 0.98

def get_action(state, epsilon = 0.05):
    if (np.random.randint(1, 101)/100 < epsilon):
        return int(np.random.choice(MATRIX_SIZE))
    return np.argmax(Q[state,:])

def update(current_state, action, gamma):
    max_index = np.argmax(Q[:,action])
    max_value = Q[max_index, action]

    Q[current_state, action] = R[current_state, action] + gamma * max_value
    print( 'max_value', R[current_state, action] + gamma * max_value )

    if (np.max( Q ) > 0):
        return (np.sum( Q / np.max( Q ) * 100 ))
    else:
        return (0)

scores = []
for i in range(1000):
    current_state = np.random.choice(MATRIX_SIZE)
    action = get_action(current_state, i/50)
    score = update(current_state, action, gamma)
    scores.append(score)


print("Trained Q matrix: ")
print(Q/np.max(Q)*100)
plt.plot(scores)
plt.show()