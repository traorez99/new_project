import numpy as np
import matplotlib.pyplot as plt
import time
from roblib import *  # https://www.ensta-bretagne.fr/jaulin/roblib.py

# Paramètres ajustés pour un contrôle optimisé
K_att, K_rep, K_v, K_w, K_theta = 20, 35, 20, 50, 15  # Forces ajustées
x = np.array([0.3, 0.3, 0.5, 0])  # Position initiale

goals = [np.array([2, 1]), np.array([1, 1.5])]  # Objectifs

d_safe = 0.15  # Distance de sécurité
obstacles = [np.array([1, 1.2]), np.array([1.8, 0.8]), np.array([2.2, 1.3]), np.array([2.7, 1.8])]  # Obstacles

walls = [np.array([[0, 0], [0, 2]]), np.array([[0, 2], [3, 2]]), np.array([[3, 2], [3, 0]]), np.array([[3, 0], [0, 0]])]  # Bords du terrain

dt, d_eps = 0.01, 0.001
xmin, xmax, ymin, ymax = 0, 3, 0, 2  # Terrain 2m x 3m
ax = init_figure(xmin, xmax, ymin, ymax)
fig = plt.gcf()

running = True
def stop(event=None):
    global running
    print("🟥 Simulation arrêtée.")
    running = False

fig.canvas.mpl_connect('close_event', stop)

# Fonction pour calculer la dynamique du robot (xdot)
def xdot(x, u):
    # x[0] et x[1] sont les positions (x, y), x[2] est la vitesse linéaire, x[3] est l'orientation
    return np.array([x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), u[0], u[1]])

# Potentiel attractif conique
def attractive_potential(x, goal, K_att):
    return -K_att * (x[:2] - goal)

# Potentiel répulsif quadratique
def repulsive_potential(x, obstacles, K_rep, d_safe):
    F_rep = np.array([0.0, 0.0])
    for obs in obstacles:
        nq = x[:2] - obs
        norm_nq = max(np.linalg.norm(nq), 1e-3)
        if norm_nq < d_safe:
            F_rep += K_rep * nq / (norm_nq**2)  # Force quadratique
        else:
            F_rep += K_rep * nq / norm_nq**3  # Répulsion plus douce en dehors de la zone de sécurité
    return F_rep

# Fonction pour calculer la distance d'un point à un segment
def point_to_segment_distance(p, a, b):
    """ Calcule la distance d'un point p au segment défini par (a, b) """
    ap, ab = p - a, b - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

# Boucle de navigation pour chaque objectif
for goal in goals:
    while np.linalg.norm(x[:2] - goal) > d_eps and running:
        clear(ax)
        distance_goal = np.linalg.norm(x[:2] - goal)
        print(f"Distance au but: {distance_goal:.3f} m")

        # Calcul des forces attractives et répulsives
        F_att = attractive_potential(x, goal, K_att)
        F_rep = repulsive_potential(x, obstacles, K_rep, d_safe)

        # Ajout de la gestion des murs pour éviter de sortir du terrain
        for wall in walls:
            d = point_to_segment_distance(x[:2], wall[0], wall[1])
            if d < d_safe:
                nq = x[:2] - wall.mean(axis=0)
                norm_nq = max(np.linalg.norm(nq), 1e-3)
                F_rep += K_rep * nq / (norm_nq**2)

        # Calcul de la vitesse et de la direction du robot
        w = F_att + F_rep
        vbar, thetabar = np.linalg.norm(w), np.arctan2(w[1], w[0])

        # Mise à jour de l'angle et de la vitesse
        delta_theta = (thetabar - x[3] + np.pi) % (2 * np.pi) - np.pi
        u_v = K_v * (vbar - x[2])
        u_theta = K_w * delta_theta

        # Appliquer les commandes
        u = np.array([u_v, u_theta])

        # Mise à jour de la position du robot
        x = x + dt * xdot(x, u)
        x[0] = np.clip(x[0], xmin, xmax)  # Limiter aux bornes x
        x[1] = np.clip(x[1], ymin, ymax)  # Limiter aux bornes y

        # Visualiser la simulation
        for obs in obstacles:
            draw_disk(ax, obs, 0.05, "magenta")
        for wall in walls:
            ax.plot(wall[:, 0], wall[:, 1], 'b', linewidth=2)
        draw_disk(ax, goal, 0.05, "green")
        draw_tank(x[[0, 1, 3]], 'red', 0.08)  # Robot plus petit
        plt.pause(0.01)  # Délai pour une meilleure fluidité

    time.sleep(0.03)  # Délai entre chaque objectif

plt.close()
