import argparse
import numpy as np
import sys
import polyconfig
import polycrystal


def write_lammps():
    return


def main():

    config = polyconfig.PolyConfig()
    data = polycrystal.Polycrystal(config)

    data.initialize_grain_centers()

    for i in range(config.grains):
        data.lattice(i)
        # data.rotate_box(i)
        # data.voronoi(i)

    data.write_grain_orientation_vtk('testvtk.vtk')
    # config.convolution()
    # config.sort_grig()
    # config.check_distance()

    # write_lammps(config)


if __name__ == "__main__":
    # config = config.Config()
    main()
    print('done!')
