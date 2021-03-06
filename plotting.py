import numpy as np
import matplotlib.pyplot as plt


def plot_neighbour_angle(neighbour_angles):
    np.savetxt("neighbour_angles.txt", neighbour_angles)
    plt.hist(neighbour_angles, bins=np.arange(0, 190, 10), density=True)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Number of bonds/Total bonds")
    plt.show()


def plot_normal_vector(ax, start_point, normal):
    """Given an axes, calculate the normal vector and plot it starting at location provided"""
    ax.quiver(start_point[0], start_point[1], start_point[2],
              normal[0], normal[1], normal[2], length=200, normalize=True, linewidths=5, color=[0, 0, 1])


def plot_plane(ax, fit, selected_point_coordinate):
    """ Plot a plane on the graph corresponding to the fit."""
    xlim = [selected_point_coordinate[0] - 50, selected_point_coordinate[0] + 50]
    ylim = [selected_point_coordinate[1] - 50, selected_point_coordinate[1] + 50]
    x, y = np.meshgrid(np.arange(xlim[0], xlim[1], 10), np.arange(ylim[0], ylim[1], 10))
    z = np.zeros(x.shape)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            z[r, c] = fit[0] * x[r, c] + fit[1] * y[r, c] + fit[2]
    ax.plot_wireframe(x, y, z, color='k')


def plot_raw_data(ax, data, central_points, local_points, inverse_mask):
    # plot data points that are part of the fit in one colour and the rest in another colour
    # ax.scatter(local_points[:, 0], local_points[:, 1], local_points[:, 2], color='b')
    # ax.scatter(central_points[:, 0], central_points[:, 1], central_points[:, 2], color='g')
    # ax.scatter(data[inverse_mask, 0], data[inverse_mask, 1], data[inverse_mask, 2], color='r', alpha=0.3)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='r', alpha=0.5)

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
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
