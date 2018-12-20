import numpy as np
import util
import pyevtk.hl as pyevtkhl

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

    def initialize_grain_centers(self, seed=123):

        # for _ in range(self.grains):
        #     grain_center = self.shift*(np.random.rand(3)*2 - 1)
        #     euler_angles = 2*np.pi*np.random.rand(3)
        #     rotation_axis = np.array([np.sin(euler_angles[1])/np.cos(euler_angles[0]),
        #                               np.sin(euler_angles[1])/np.sin(euler_angles[0]),
        #                               1/np.cos(euler_angles[1])])

        #     self.grains.append(Grain(center=grain_center,
        #                              rotvt=util.rotate_rodrigues(-grain_center, rotation_axis, -euler_angles[2]),
        #                              angle=euler_angles))

        self.config.grains = 25  # REMOVE AFTER TESTING

        np.random.seed(seed=seed)
        self.grain_centers = self.shift*(np.random.rand(self.config.grains, 3)*2 - 1)
        self.euler_angles = 2*np.pi*np.random.rand(self.config.grains, 3)
        self.rotation_axes = np.zeros((self.config.grains, 3))
        self.rotation_axes[:, 0] = np.sin(self.euler_angles[:, 1])/np.cos(self.euler_angles[:, 0])
        self.rotation_axes[:, 1] = np.sin(self.euler_angles[:, 1])/np.sin(self.euler_angles[:, 0])
        self.rotation_axes[:, 2] = 1/np.cos(self.euler_angles[:, 1])
        self.rotvt = util.rotate_rodrigues(-self.grain_centers, self.rotation_axes, -self.euler_angles[:, 2])

        return

    def generate_lattice(self, ig):

        n_unit_cells = (self.config.size/self.config.lattice_constants).astype(int)
        n_possible_atoms = np.prod(n_unit_cells)*self.config.basis.shape[0]
        lattice = np.zeros((n_possible_atoms, 3))

        ia = 0  # Atom count

        # for i in range(n_unit_cells[0]):
        #     for j in range(n_unit_cells[1]):
        #         for k in range(n_unit_cells[2]):
        #             for vec in self.config.basis:
        #                 lattice[ia] = np.array([i, j, k])*self.config.lattice_constants + vec




        return voronoi_decimate(lattice, self.grain_centers)

    def write_grain_orientation_vtk(self, fname):
        print(f'Writing grain orientations: {fname}')
        # pyevtkhl.pointsToVTK(fname,
        #                      self.grain_centers[:, 0],
        #                      self.grain_centers[:, 1],
        #                      self.grain_centers[:, 2],
        #                      data={'rotvt': self.rotvt})

        with open(fname, 'w') as f:
            f.write('# vtk DataFile Version 2.0\n')
            f.write('\n')
            f.write('ASCII\n')
            f.write('DATASET UNSTRUCTURED_GRID\n')
            f.write(f'POINTS {self.config.grains} float\n')
            for loc in self.grain_centers:
                f.write(f'{loc[0]} {loc[1]} {loc[2]}\n')
            f.write(f'POINT_DATA {self.config.grains}\n')
            f.write('VECTORS grain_orientations float\n')
            for vec in self.rotvt:
                f.write(f'{vec[0]} {vec[1]} {vec[2]}\n')

        print('Finished writing grain orientations.')

        return