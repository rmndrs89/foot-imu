import numpy as np

def axangle2mat(u, theta, is_normalized=False):
    if not is_normalized:
        u /= np.linalg.norm(u)
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    return np.array([
        [u[0]*u[0]*C+c, u[0]*u[1]*C-u[2]*s, u[0]*u[2]*C+u[1]*s],
        [u[1]*u[0]*C+u[2]*s, u[1]*u[1]*C+c, u[1]*u[2]*C-u[0]*s],
        [u[2]*u[0]*C-u[1]*s, u[2]*u[1]*C+u[0]*s, u[2]*u[2]*C+c]
    ])