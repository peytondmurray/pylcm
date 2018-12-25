import numpy as np
import util
import pyevtk.hl as pyevtkhl


# TODO Flesh out the rest of Lattice dunders
class Lattice:

    def __init__(self, lattice=None, elements=None, n_atoms=None):
        if isinstance(lattice, np.ndarray) and len(lattice.shape) == 2 and lattice.shape[1] == 3:
            if isinstance(elements, np.ndarray):
                if lattice.shape[0] == elements.shape[0]:
                    self.lattice = lattice
                    self.elements = elements
                else:
                    raise ValueError('Input element and lattice arrays must contain same number of atoms.')
            else:
                raise ValueError('Elements argument must be of type numpy.ndarray')
        elif lattice is None and elements is None and n_atoms is not None:
            self.lattice = np.empty((n_atoms, 3), dtype=float)
            self.elements = np.empty((n_atoms), dtype=str)
        else:
            raise ValueError('Invalid arguments.')
        return

    def size(self):
        return self.lattice.shape[0]

    def __getitem__(self, index):
        return self.lattice[index], self.elements[index]

    def __setitem__(self, index, item):
        self.lattice[index] = item[0]
        self.elements[index] = item[1]
        return

    def __add__(self, item):
        if isinstance(item, Lattice):
            return self.add_lattice(item)
        elif isinstance(item, tuple) or isinstance(item, list):
            return self.add_arrays(item)

    def add_lattice(self, other):
        return Lattice(np.vstack((self.lattice, other.lattice)), np.vstack((self.elements, other.elements)))

    def add_arrays(self, item):
        return Lattice(np.vstack((self.lattice, item[0])), np.vstack((self.elements + item[1])))

    def __len__(self):
        return self.size()

    def __iter__(self):
        pass

    def __hash__(self):
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def __mul__(self, factor):
        return Lattice(self.lattice*factor, self.elements)

    def __div__(self, factor):
        return self*(1/factor)

    def __neg__(self):
        return self*-1

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return Lattice(self.lattice@other, self.elements)
        else:
            raise ValueError('Lattice can only __matmul__ with an numpy.ndarray')


class Polycrystal:

    def __init__(self, config):

        self.config = config
        self.shift = np.array([0.5, 0.5, 0.5])  # ?????
        self.grains = []
        self.lattice = None
        return

    def initialize_grain_centers(self, seed=123):
        """Generate random grain centers. self.grain_centers will be set to an np.ndarray of shape
        (self.config.grains, 3), with each row containing the x, y, z coordinates of each grain center.

        Parameters
        ----------
        seed : int, optional
            Random seed. Default is 123, no need to change unless you really want.

        """

        self.config.grains = 25  # REMOVE AFTER TESTING

        np.random.seed(seed=seed)
        self.grain_centers = np.random.rand(self.config.grains, 3)

        return

    def generate_lattice(self, ig: int, basis: dict):
        """Generate a lattice centered on self.grain_centers[ig], using the given basis vectors.

        Parameters
        ----------
        ig : int
            grain index
        basis : dict
            Dict of the form {element: 3-tuple}. Contains the element names of the atoms located at the (x, y, z)
            locations given in the respective 3-tuples.

        """

        relative_loc = self.grain_centers[ig]/self.config.size          # Relative location in sim box [0, 1]

        # Number of unit cells span simulation box
        n_uc_span = util.n_uc_fill(self.config.size, self.config.lattice_vectors, kind='project')
        n_uc = (relative_loc*n_uc_span).astype(int)                     # Location of grain in number of unit cells

        _lattice = np.empty((n_uc_span*basis.shape[0], 3), dtype=float)
        _elements = np.empty((n_uc_span*basis.shape[0], 1), dtype=str)

        _lattice = Lattice(shape=)

        ia = 0  # Atom count
        for i in range(-n_uc[0], n_uc_span[0] - n_uc[0]):
            for j in range(-n_uc[1], n_uc_span[1] - n_uc[1]):
                for k in range(-n_uc[2], n_uc_span[2] - n_uc[2]):
                    for el, vec in basis.items():
                        _lattice[ia] = self.grain_centers[ig] + np.array([i, j, k])*self.config.lattice_constants + vec
                        _elements[ia] = el
                        ia += 1

        _lattice, _elements = self.voronoi_decimate(self.grains, _lattice, _elements)

        if self.lattice is not None:
            self.lattice = self.lattice + Lattice(_lattice, _elements)
        else:
            self.lattice = Lattice(_lattice, _elements)

        return

    def voronoi_decimate(self, ig: int, lattice: Lattice) -> Lattice:
        """Given a grain index and input Lattice instance, this function checks which atoms in the lattice
        actually belong to the grain centered at self.grain_centers[ig]. It does this by voronoi tesselation; any
        atoms which are closer to a grain center other than self.grain_centers[ig] are not returned.

        Parameters
        ----------
        ig : int
            Index of the grain associated with the given input lattice and element arrays.
        lattice : np.ndarray
            Lattice which is to be tesselated and decimated.

        Returns
        -------
        Lattice
            Contains the locations and names (i.e. chemical symbols) of the atoms which are closer to
            self.grain_centers[ig] than any other grain center

        """


        vvalid_lattice = Lattice()

        for i in range(lattice.shape[0]):
            if self.nearest_grain(lattice[i]) == ig:
                valid_lattice += [lattice[i]]
                valid_elements += [elements[i]]

        return np.array(valid_lattice), np.array(valid_elements)

    def nearest_grain(self, vec: np.ndarray) -> int:
        """Find the index of the grain center nearest to vec.

        Parameters
        ----------
        vec : np.ndarray
            Input vector of shape (3,) or (1, 3).

        Returns
        -------
        int
            Index of grain center nearest to vec.
        """

        return np.argmin(np.sum(np.square(self.grain_centers - vec), axis=1))

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
