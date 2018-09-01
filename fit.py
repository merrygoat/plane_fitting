import matplotlib.pyplot as plt
import numpy as np
import plotting


def main(do_plot_plane=1, do_plot__normal_vectors=1):
    data = np.loadtxt("example_sheet.xyz", skiprows=2, usecols=(1, 2, 3))

    central_points = get_central_points(data)

    for point_number in central_points:
        selected_point_coordinate = data[point_number]
        inverse_mask, neighbour_list = get_nearest_neighbours(data, selected_point_coordinate)
        ax = plotting.plot_raw_data(data, neighbour_list, inverse_mask)
        fit = do_plane_fit(neighbour_list)
        normal = get_plane_normal(selected_point_coordinate, fit)
        if do_plot_plane:
            plotting.plot_plane(ax, fit, selected_point_coordinate)
        if do_plot__normal_vectors:
            plotting.plot_normal_vector(ax, selected_point_coordinate, normal)
        if do_plot_plane or do_plot__normal_vectors:
            plotting.set_equal_axes(ax, data)
            plt.show()


def get_central_points(data):
    """Fitting will be odd around the edges so select the points in the middle of the sheet"""
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    y_mask = y.min + 50 < data[:, 1] < y.max - 50
    z_mask = z.min + 30 < data[:, 2] < z.max - 30
    mask = np.logical_and(y_mask, z_mask)
    return data[mask, :]

def get_plane_normal(start_point, fit):
    """Calculate normal vector at one point on the plane"""
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
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    # First convert to unit vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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


def get_nearest_neighbours(data, selected_point_coordinate):
    """ Given a point in the dataset, find its nearest neighbours """

    xdistances = np.abs(selected_point_coordinate[0] - data[:, 0])
    ydistances = np.abs(selected_point_coordinate[1] - data[:, 1])
    zdistances = np.abs(selected_point_coordinate[2] - data[:, 2])
    xmask = xdistances < 30
    ymask = ydistances < 30
    zmask = zdistances < 100
    distance_mask = np.logical_and(np.logical_and(xmask, ymask), zmask)
    inverse_mask = np.logical_not(distance_mask)
    local_points = data[distance_mask, :]

    return inverse_mask, local_points


main()
