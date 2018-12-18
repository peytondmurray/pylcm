import numpy as np


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
        # self.shift = np.array([0.5, 0.5, 0.5])  # ?????
        self.grains = []
        return

    def initialize_grain_centers(self):
        
        # The hell is this doing?
        for i in range(self.grains):
            center = self.shift*(np.random.rand(3)*2 - 1)
            angle = 2*np.pi*np.random.rand(3)
            rotv = np.array([np.sin(angle[1])/np.cos(angle[0]),
                             np.sin(angle[1])/np.sin(angle[0]),
                             1/np.cos(angle[1])
            self.grains.append(Grain(center,
                                     


        return