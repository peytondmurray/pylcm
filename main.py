import argparse
import numpy as np
import sys
import configuration
import polycrystal


def write_lammps():
    return


def main():

    config = configuration.Config()
    data = polycrystal.Polycrystal(config)

    data.initialize_grain_centers()
    data.write_grains()

    for i in range(config.grains):
        config.lattice(i)
        config.rotate_box(i)
        config.voronoi(i)
    
    config.convolution()
    config.sort_grig()
    config.check_distance()

    write_lammps(config)


if __name__ == "__main__":
    config = config.Config()
    print('done!')