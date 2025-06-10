

import numpy as np
from rob import *  # https://www.ensta-bretagne.fr/jaulin/roblib.py


def xdot(x, u):
    return np.array([x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), u[0], u[1]])


def f1(x1, x2):  
    return -x1, -x2


K_att = 2; K_rep = 10
x = np.array([0, 0, 1, 0])  # x, y, v, θ
vhat = np.array([1, 1])
phat = np.array([8, 8]) #potentiel attractif (goal)
qhat = np.array([3, 3]) #potentiel répulsif (obstacle)
dt = 0.005
xmin = 0; xmax = 10; ymin = 0; ymax =10
ax = init_figure(xmin ,xmax, ymin, ymax)


d_eps = 0.2; #Erreur de position acceptable


while np.linalg.norm(x[:2] - phat) > d_eps :
   
    clear(ax)
    print(np.linalg.norm(x[:2] - phat))
    nq = x[:2] - qhat


    w = vhat - 2 * K_att * (x[:2] - phat) + K_rep * nq / (np.linalg.norm(nq)**3)


    vbar = np.linalg.norm(w)
    thetabar = np.arctan2(w[1], w[0])


    u = 20*np.array([vbar - x[2], 2 * np.arctan(np.tan(0.5 * (thetabar - x[3])))]) #u = np.array([vbar - x[2], 2 * VALEUR A CHANGER * np.arctan(np.tan(0.5 * (thetabar - x[3])))])


    x = x + dt * xdot(x, u)  


    draw_disk(ax, qhat, 0.3, "magenta")
    draw_disk(ax, phat, 0.2, "green")
    draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x, y, θ
    draw_field(phat, qhat, vhat, 0.2, xmin, xmax, ymin, ymax)


plt.pause(1)
#Pour la prochaine fois
#Trouver les coeff sur lesquels on peut agir
#Pourquoi le champ du point répulsif de bouge pas
#Améliorer la précision
