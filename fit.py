import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotting


def main(do_plot_plane=1, do_plot__normal_vectors=1):
    data = np.loadtxt("example_sheet.xyz", skiprows=2, usecols=(1, 2, 3))

    for point_number in [426, 177, 333, 415, 457, 533]:

        selected_point_coordinate = data[point_number]
        inverse_mask, neighbour_list = get_nearest_neighbours(data, selected_point_coordinate)
        ax = plotting.plot_raw_data(data, neighbour_list, inverse_mask)
        fit = do_plane_fit(neighbour_list)
        if do_plot_plane:
            plotting.plot_plane(ax, fit, selected_point_coordinate)
        if do_plot__normal_vectors:
            plotting.plot_normal_vector(ax, fit, selected_point_coordinate)
        if do_plot_plane or do_plot__normal_vectors:
            plt.show()
        neighbour_vectors = get_neighbour_vectors(selected_point_coordinate, neighbour_list)


def get_neighbour_vectors(particle_coordinates, neighbour_coordinates):
    """ Get the vectors between a particle and its neighbours """
    return neighbour_coordinates - particle_coordinates


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def do_plane_fit(local_points):
    """ Fit a flat plane to a particle and its nearest neighbours. This code shamelessly ripped from stack exchange """
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
    # Given a point in the dataset, find its nearest neighbours

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
