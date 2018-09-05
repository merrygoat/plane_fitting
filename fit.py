import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotting


def main(do_plot_plane=1, do_plot__normal_vectors=1, do_plot_raw_data=1):
    data = np.loadtxt("example_sheet.xyz", skiprows=2, usecols=(1, 2, 3))

    central_points = get_central_points(data)
    neighbour_angles = []
    vector_ax = plt.subplot(111, projection='3d')
    for selected_point_coordinate in central_points:
        inverse_mask, plane_neighbour_list = get_nearest_neighbours(data, selected_point_coordinate, [30, 30, 100])
        fit = do_plane_fit(plane_neighbour_list)
        normal = get_plane_normal(selected_point_coordinate, fit)
        _, bond_neighbour_list = get_nearest_neighbours(data, selected_point_coordinate, [20, 20, 20])
        for particle in bond_neighbour_list:
            if np.any(selected_point_coordinate != particle):
                neighbour_vector = selected_point_coordinate - particle
                neighbour_angles.append(angle_between(neighbour_vector, normal))
        # Plotting
        if do_plot_raw_data:
            plotting.plot_raw_data(vector_ax, data, central_points, plane_neighbour_list, inverse_mask)
        if do_plot_plane:
            plotting.plot_plane(vector_ax, fit, selected_point_coordinate)
        if do_plot__normal_vectors:
            plotting.plot_normal_vector(vector_ax, selected_point_coordinate, normal)
        if do_plot_plane or do_plot_raw_data:
            plotting.set_equal_axes(vector_ax, data)
            plt.show()
        plotting.set_equal_axes(vector_ax, data)
    plt.show()

    plotting.plot_neighbour_angle(neighbour_angles)


def get_central_points(data):
    """Fitting will be odd around the edges so select the points in the middle of the sheet"""
    x_edge_cut = 0
    y_edge_cut = 20
    z_edge_cut = 10
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    x_mask = np.logical_and(data[:, 0] > (x.min() + x_edge_cut), data[:, 0] < (x.max() - x_edge_cut))
    y_mask = np.logical_and(data[:, 1] > (y.min() + y_edge_cut), data[:, 1] < (y.max() - y_edge_cut))
    z_mask = np.logical_and(data[:, 2] > (z.min() + z_edge_cut), data[:, 2] < (z.max() - z_edge_cut))
    mask = np.logical_and.reduce(np.array([x_mask, y_mask, z_mask]))
    return data[mask, :]


def get_plane_normal(start_point, fit):
    """Calculate normal vector of a plane described by the coefficients in fit"""
    point_1 = np.array([start_point[0],
                        start_point[1],
                        fit[0] * start_point[0] + fit[1] * start_point[1] + fit[2]])
    point_2 = np.array([start_point[0] + 50,
                        start_point[1],
                        fit[0] * (start_point[0] + 50) + fit[1] * start_point[1] + fit[2]])
    point_3 = np.array([start_point[0],
                        start_point[1] + 50,
                        fit[0] * start_point[0] + fit[1] * (start_point[1] + 50) + fit[2]])
    vector_1 = point_2 - point_1
    vector_2 = point_3 - point_1
    normal = np.cross(vector_1, vector_2)
    return normal


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    # First convert to unit vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def do_plane_fit(local_points):
    """ Fit a flat plane to a particle and its nearest neighbours using a least squares fit.

    This code shamelessly ripped from stack exchange:
    https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points/20700063#20700063
    """

    tmp_A = []
    tmp_b = []
    for i in range(len(local_points[:, 0])):
        tmp_A.append([local_points[:, 0][i], local_points[:, 1][i], 1])
        tmp_b.append(local_points[:, 2][i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit_coefficients = (A.T * A).I * A.T * b

    return fit_coefficients


def get_nearest_neighbours(data, selected_point_coordinate, distances):
    """ Given a point in the dataset, find its nearest neighbours """

    xdistances = np.abs(selected_point_coordinate[0] - data[:, 0])
    ydistances = np.abs(selected_point_coordinate[1] - data[:, 1])
    zdistances = np.abs(selected_point_coordinate[2] - data[:, 2])
    xmask = xdistances < distances[0]
    ymask = ydistances < distances[1]
    zmask = zdistances < distances[2]
    distance_mask = np.logical_and(np.logical_and(xmask, ymask), zmask)
    inverse_mask = np.logical_not(distance_mask)
    local_points = data[distance_mask, :]

    return inverse_mask, local_points


main()
