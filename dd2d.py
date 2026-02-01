import numpy as np

def iterate(x, dt, u):
    """
    Diff-drive process model
    x : np.array shape (3,) -> [x, y, theta]
    u : [v, omega]
    dt: timestep
    """
    v, omega = u
    theta = x[2]

    x_next = np.zeros(3)
    x_next[0] = x[0] + v * np.cos(theta) * dt
    x_next[1] = x[1] + v * np.sin(theta) * dt
    x_next[2] = x[2] + omega * dt

    return x_next
