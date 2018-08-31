import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_normal_vector(ax, vector, start_point):
    """Given an axes, plot a vector on it starting at location provided"""
    ax.quiver(start_point[0], start_point[1], start_point[2],
              vector[0], vector[1], -1, length=200, normalize=True, linewidths=5)


def plot_plane(ax, fit, selected_point_coordinate):
    """ Plot a plane on the graph corresponding to the fit."""
    xlim = [selected_point_coordinate[0] - 50, selected_point_coordinate[0] + 50]
    ylim = [selected_point_coordinate[1] - 50, selected_point_coordinate[1] + 50]
    x, y = np.meshgrid(np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1]))
    z = np.zeros(x.shape)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            z[r, c] = fit[0] * x[r, c] + fit[1] * y[r, c] + fit[2]
    ax.plot_wireframe(x, y, z, color='k')


def plot_raw_data(data, local_points, inverse_mask):
    # plot data points that are part of the fit in one colour and the rest in another colour
    plt.figure()
    plt.axis('equal')
    ax = plt.subplot(111, projection='3d')
    ax.scatter(local_points[:, 0], local_points[:, 1], local_points[:, 2], color='b')
    ax.scatter(data[inverse_mask, 0], data[inverse_mask, 1], data[inverse_mask, 2], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax
