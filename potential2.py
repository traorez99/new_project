import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def draw_field(phat, qhat, vhat, a, xmin, xmax, ymin, ymax):
    X    = arange(xmin,xmax,a)
    Y    = arange(ymin,ymax,a)
    P1,P2 = meshgrid(X,Y)
    Nq1 = (P1 - qhat[0])
    Nq2 = (P2 - qhat[1])
    VX = vhat[0] - 2 * (P1 - phat[0]) + Nq1 / (Nq1**2 + Nq2**2)**(3/2)
    VY = vhat[1] - 2 * (P2 - phat[1]) + Nq2 / (Nq1**2 + Nq2**2)**(3/2)
    quiver(X,Y,VX,VY) 

def add1(M):
    return np.vstack((M, np.ones((1, M.shape[1]))))

def tran2H(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

def rot2H(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def plot2D(M, col='darkblue', w=2):
    plt.plot(M[0, :], M[1, :], col, linewidth=w)

def clear(ax):
    ax.cla()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def draw_tank(x, col='darkblue', r=1, w=2):
    mx, my, theta = x[0], x[1], x[2]
    M = r * np.array([[1, -1, 0, 0, -1, -1, 0, 0, -1, 1, 0, 0, 3, 3, 0],
                      [-2, -2, -2, -1, -1, 1, 1, 2, 2, 2, 2, 1, 0.5, -0.5, -1]])
    M = add1(M)
    plot2D(tran2H(mx, my) @ rot2H(theta) @ M, col, w)

def init_figure(xmin, xmax, ymin, ymax):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax

def draw_disk(ax, c, r, col, alph=0.7, w=1):
    e = Ellipse(xy=c, width=2*r, height=2*r, angle=0, linewidth=w)
    ax.add_patch(e)
    e.set_alpha(alph)
    e.set_facecolor(col)

def xdot(x, u):
    return np.array([x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), u[0], u[1]])

K_att, K_rep, K_v, K_w, K_theta = 7, 10, 25, 30, 8
x = np.array([0, 0, 1, 0])
phat = np.array([9, 9])
obstacles = [np.array([8, 5]), np.array([4.8, 5.2]), np.array([6, 7.5]), np.array([1, 8.1])]

dt = 0.009
xmin, xmax, ymin, ymax = 0, 10, 0, 10
d_eps = 0.05

ax = init_figure(xmin, xmax, ymin, ymax)
fig = plt.gcf()

running = True
def stop(event=None):
    global running
    print("🟥 Simulation arrêtée.")
    running = False
fig.canvas.mpl_connect('close_event', stop)

while np.linalg.norm(x[:2] - phat) > d_eps and running:
    clear(ax)
    distance_goal = np.linalg.norm(x[:2] - phat)
    print(f"📍 Distance à l'objectif : {distance_goal:.6f}") 
    
    F_att = -K_att * (x[:2] - phat) / max(distance_goal, 1e-3)
    
    F_rep = np.array([0.0, 0.0])
    d_safe = 1.0
    for qhat in obstacles:
        nq = x[:2] - qhat
        norm_nq = max(np.linalg.norm(nq), 1e-3)
        if norm_nq < d_safe:
            F_rep += K_rep * nq / (norm_nq**2)
        else:
            F_rep += K_rep * nq / norm_nq**3
    
    w = F_att + F_rep
    vbar, thetabar = np.linalg.norm(w), np.arctan2(w[1], w[0])
    delta_theta = (thetabar - x[3] + np.pi) % (2 * np.pi) - np.pi
    erreur_theta = np.clip(delta_theta ** 2, 0, 5)
    u = np.array([K_v * (vbar - x[2]), 2 * K_w * np.arctan(np.tan(0.5 * delta_theta)) + K_theta * erreur_theta])
    
    x = x + dt * xdot(x, u)
    
    for qhat in obstacles:
        draw_disk(ax, qhat, 0.3, "magenta")
    draw_disk(ax, phat, 0.2, "green")
    draw_tank(x[[0, 1, 3]], 'red', 0.2)
    plt.pause(0.001)

plt.close()
