import numpy as np
import pandas as pd
import util
import pyevtk.hl as pyevtkhl


# class Lattice:

#     def __init__(self, lattice=None, elements=None, n_atoms=None):
#         if isinstance(lattice, np.ndarray) and len(lattice.shape) == 2 and lattice.shape[1] == 3:
#             if isinstance(elements, np.ndarray):
#                 if lattice.shape[0] == elements.shape[0]:
#                     self.lattice = lattice
#                     self.elements = elements
#                 else:
#                     raise ValueError('Input element and lattice arrays must contain same number of atoms.')
#             else:
#                 raise ValueError('Elements argument must be of type numpy.ndarray')
#         elif lattice is None and elements is None and n_atoms is not None:
#             self.lattice = np.empty((n_atoms, 3), dtype=float)
#             self.elements = np.empty((n_atoms), dtype=str)
#         else:
#             raise ValueError('Invalid arguments.')
#         return

#     def size(self):
#         return self.lattice.shape[0]

#     def __getitem__(self, index):
#         return self.lattice[index], self.elements[index]

#     def __setitem__(self, index, item):
#         self.lattice[index] = item[0]
#         self.elements[index] = item[1]
#         return

#     def __add__(self, item):
#         if isinstance(item, Lattice):
#             return self.add_lattice(item)
#         elif isinstance(item, tuple) or isinstance(item, list):
#             return self.add_arrays(item[0], item[1])

#     def add_lattice(self, other):
#         return Lattice(np.vstack((self.lattice, other.lattice)), np.vstack((self.elements, other.elements)))

#     def add_arrays(self, lattice, elements):
#         return Lattice(np.vstack((self.lattice, lattice)), np.vstack((self.elements + elements)))

#     def __len__(self):
#         return self.size()

#     def __hash__(self):
#         pass

#     def __repr__(self):
#         pass

#     def __str__(self):
#         pass

#     def __mul__(self, factor):
#         return Lattice(self.lattice*factor, self.elements)

#     def __div__(self, factor):
#         return self*(1/factor)

#     def __neg__(self):
#         return self*-1

#     def __matmul__(self, other):
#         if isinstance(other, np.ndarray):
#             return Lattice(self.lattice@other, self.elements)
#         else:
#             raise ValueError('Lattice can only __matmul__ with an numpy.ndarray')

#     def __iter__(self):
#         return self

#     def __next__(self):
#         g = ((vec, el) for vec, el in zip(self.lattice, self.elements))
#         yield from g


class Grain:

    def __init__(self, center, lattice_vectors, basis, axis, angle=0):
        self.center = center
        self.axis = axis
        self.angle = angle
        self.basis = basis
        self.atoms = np.empty(0, dtype=float)
        self.elements = np.empty(0, dtype=str)
        self.lattice_vectors = lattice_vectors
        self.rotated_basis = {el: util.rotate_rodrigues(vec, self.axis, self.angle) for el, vec in self.basis.items()}
        self.rotated_lattice_vectors = util.rotate_rodrigues(self.lattice_vectors, self.axis, self.angle)
        return

    def fill_lattice_region(self, size: np.ndarray):
        """Fill the part of the simulation region with atoms.

        Parameters
        ----------
        size : np.ndarray
            3-element array containing [xsize, ysize, zsize], the sizes of the simulation region.

        """

        # relative_loc = self.center/size

        _atoms = np.empty((n_atoms_generated, 3), dtype=float)      # Arrays to store atoms locations and element names
        _elements = np.empty((n_atoms_generated, 1), dtype=str)

        ia = 0                                                      # Atom count
        for i in range(-n_uc[0], n_uc_span[0] - n_uc[0]):
            for j in range(-n_uc[1], n_uc_span[1] - n_uc[1]):
                for k in range(-n_uc[2], n_uc_span[2] - n_uc[2]):
                    for el, vec in self.rotated_basis.items():
                        _atoms[ia] = self.center+np.array([i, j, k])*self.rotated_lattice_vectors+vec
                        _elements[ia] = el
                        ia += 1

        return


class Polycrystal:

    def __init__(self, config):

        self.config = config
        self.lattice = None
        self.grains = self.initialize_grain_centers()

        return

    def initialize_grain_centers(self, seed=123, randomize_basis=True):
        """Generate random grain centers. self.grain_centers will be set to an np.ndarray of shape
        (self.config.grains, 3), with each row containing the x, y, z coordinates of each grain center.

        Parameters
        ----------
        seed : int, optional
            Random seed. Default is 123, no need to change unless you really want.

        """

        np.random.seed(seed=seed)
        grain_centers = np.random.rand(self.config.ngrains, 3)*self.config.size
        grain_vectors = util.generate_random_vectors(self.config.ngrains)
        grain_angles = np.random.rand(self.config.ngrains, 1)*2*np.pi

        grains = []

        for i in range(self.config.ngrains):
            grains.append(Grain(grain_centers[i],
                                self.config.lattice_vectors,
                                self.config.basis,
                                grain_vectors[i],
                                grain_angles[i]))

        return grains

    def _voronoi_decimate(self, ig: int, lattice: Lattice) -> Lattice:
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

        valid_lattice = Lattice()

        for i in range(lattice.size()):
            if self.nearest_grain(lattice[i][0]) == ig:
                valid_lattice.add_arrays(*lattice[i])

        return valid_lattice

    def generate_lattice(self, ig: int):

        relative_loc = self.grains[ig].center/self.config.size  # Relative location of grain center in sim box [0, 1]

        # Number of unit cells span simulation box
        n_uc_span = util.n_uc_fill(self.config.size,
                                   self.grains[ig].rotated_lattice_vectors,
                                   kind='project')
        n_uc = (relative_loc*n_uc_span).astype(int)             # Location of grain in number of unit cells

        ia = 0  # Atom count
        for i in range(-n_uc[0], n_uc_span[0] - n_uc[0]):
            for j in range(-n_uc[1], n_uc_span[1] - n_uc[1]):
                for k in range(-n_uc[2], n_uc_span[2] - n_uc[2]):
                    for el, vec in self.grains[ig].rotated_basis.items():
                        _lattice[ia] = (self.grains[ig].center+np.array([i, j, k])*self.grains[ig].lattice_vectors+vec,
                                        el)
                        ia += 1

        return


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

        return np.argmin(np.sum(np.square(self.get_grain_centers() - vec), axis=1))

    def get_grain_centers(self):
        c = np.empty((self.config.ngrains, 3))
        for i in range(self.config.ngrains):
            c[i] = self.grains[i].center
        return c

    def get_grain_axes(self):
        a = np.empty((self.config.ngrains, 3))
        for i in range(self.config.ngrains):
            a[i] = self.grains[i].axes
        return a

    def get_grain_angles(self):
        a = np.empty((self.config.ngrains, 3))
        for i in range(self.config.ngrains):
            a[i] = self.grains[i].angle
        return a

    def __repr__(self):
        print(str(self))
        return

    def __str__(self):
        df = pd.DataFrame(data=np.hstack((self.get_grain_centers(), self.get_grain_axes(), self.get_grain_angles())),
                          columns=['grain_x', 'grain_y', 'grain_z', 'axis_x', 'axis_y', 'axis_z', 'angle'])

        config_str = '\n\t'.join(str(self.config).split('\n'))

        ret = 'Polycrystal()\n'
        ret += f'\n\t{config_str}\n\n'
        ret += 'Grains:\n'
        ret += str(df)+'\n\n'
        ret += 'Grain Bases:\n'
        for i in range(self.config.ngrains):
            ret += f'{i}\n'
            for el, vec in self.grains[i].basis:
                ret += f'\t{el}\t{vec}\n'

        return ret