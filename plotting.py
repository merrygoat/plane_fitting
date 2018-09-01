import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_normal_vector(ax, fit, start_point):
    """Given an axes, plot a vector on it starting at location provided"""
    point_1 = np.array([start_point[0], start_point[1], fit[0] * start_point[0] + fit[1] * start_point[1] + fit[2]])
    point_2 = np.array([start_point[0] + 50, start_point[1], fit[0] * (start_point[0] + 50) + fit[1] * start_point[1] + fit[2]])
    point_3 = np.array([start_point[0], start_point[1] + 50, fit[0] * start_point[0] + fit[1] * (start_point[1] + 50) + fit[2]])
    vector_1 = point_2 - point_1
    vector_2 = point_3 - point_1
    normal = np.cross(vector_1, vector_2)

    #ax.scatter(point_1[0], point_1[1], point_1[2], color='g', s=500)
    #ax.scatter(point_2[0], point_2[1], point_2[2], color='g', s=500)
    #ax.scatter(point_3[0], point_3[1], point_3[2], color='g', s=500)

    #ax.quiver(start_point[0], start_point[1], start_point[2],
    #          vector_1[0], vector_1[1], vector_1[2], length=200, normalize=True, linewidths=5, color=[1, 0, 0])
    #ax.quiver(start_point[0], start_point[1], start_point[2],
    #          vector_2[0], vector_2[1], vector_2[2], length=200, normalize=True, linewidths=5, color=[0, 1, 0])
    ax.quiver(start_point[0], start_point[1], start_point[2],
              normal[0], normal[1], normal[2], length=200, normalize=True, linewidths=5, color=[0, 0, 1])


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
    ax = plt.subplot(111, projection='3d')
    ax.scatter(local_points[:, 0], local_points[:, 1], local_points[:, 2], color='b')
    ax.scatter(data[inverse_mask, 0], data[inverse_mask, 1], data[inverse_mask, 2], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax


def set_equal_axes(ax, data):
    """ Matplotlib does not set equal scale 3D axes correctly so it must be done manually"""
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
