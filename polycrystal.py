import numpy as np
import util


class Grain:

    def __init__(self, center: np.ndarray, rotvt: np.ndarray, angle: np.ndarray):
        self.center = center    # Grain center
        self.rotvt = rotvt      # Inverted rotational vector
        self.angle = angle      # Grain rotation euler angles
        return


class Grid:

    def __init__(self, ids: np.ndarray, r: np.ndarray):
        self.ids = ids          # Atom ids
        self.r = r              # ???
        return


class Polycrystal:

    def __init__(self, config):

        self.config = config
        self.shift = np.array([0.5, 0.5, 0.5])  # ?????
        self.grains = []
        return

    def initialize_grain_centers(self, seed):

        # for _ in range(self.grains):
        #     grain_center = self.shift*(np.random.rand(3)*2 - 1)
        #     euler_angles = 2*np.pi*np.random.rand(3)
        #     rotation_axis = np.array([np.sin(euler_angles[1])/np.cos(euler_angles[0]),
        #                               np.sin(euler_angles[1])/np.sin(euler_angles[0]),
        #                               1/np.cos(euler_angles[1])])

        #     self.grains.append(Grain(center=grain_center,
        #                              rotvt=util.rotate_rodrigues(-grain_center, rotation_axis, -euler_angles[2]),
        #                              angle=euler_angles))

        self.grain_centers = self.shift*(np.random.rand(self.n_grains, 3)*2 - 1)
        self.euler_angles = 2*np.pi*np.random.rand(self.n_grains, 3)
        self.rotation_axes = np.array([np.sin(self.euler_angles[:, 1])/np.cos(self.euler_angles[:, 0]),
                                       np.sin(self.euler_angles[:, 1])/np.sin(self.euler_angles[:, 0]),
                                       1/np.cos(self.euler_angles[:, 1])])
        self.rotvt = util.rotate_rodrigues(-self.grain_centers, self.rotation_axes, -self.euler_angles[:, 2])

        return

    def lattice(self):
        N = (self.atoms_grain/self.unit_cell.cols())**1/3
        return


    def write_grain_orientation_vtk(self, fname):
        with open(fname, 'w') as f: