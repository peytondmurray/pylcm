import argparse
import numpy as np
import sys


def write_lammps():
    return


def generate_parser():
    parser = argparse.ArgumentParser(description='Generate polycrystal files for input into LAMMPs.')
    parser.add_argument('-v', '--volume',
                        type=float,
                        required=False,
                        metavar='VALUE',
                        dest='volume',
                        default=np.nan,
                        help='Volume per atom')
    parser.add_argument('-g', '--grains',
                        type=int,
                        required=False,
                        metavar='INT',
                        dest='grains',
                        default=1,
                        help='Number of grains')
    parser.add_argument('-a' '--width',
                        type=float,
                        required=False,
                        metavar='VALUE',
                        dest='a',
                        default=10,
                        help='Grain width [nm]')
    parser.add_argument('-b', '--height',
                        type=float,
                        required=False,
                        metavar='VALUE',
                        dest='b',
                        default=10,
                        help='Grain height [nm]')
    parser.add_argument('-c', '--length',
                        type=float,
                        required=False,
                        metavar='VALUE',
                        dest='c',
                        default=10,
                        help='Grain length [nm]')
    parser.add_argument('-t', '--type',
                        type=str,
                        required=False,
                        metavar='STRUCTURE',
                        dest='type',
                        default='fcc',
                        help='Lattice structure')
    parser.add_argument('-o', '--out',
                        type=str,
                        required=False,
                        metavar='FILETYPE',
                        dest='out',
                        default='lmp',
                        help='Output filetype')
    parser.add_argument('-i', '--input',
                        required=False,
                        metavar='VALUE',
                        dest='input',
                        default=False,
                        help='Input grain parameters file')
    parser.add_argument('-e', '--element',
                        type=str,
                        required=False,
                        metavar='ELEMENT',
                        dest='element',
                        default='Xe',
                        help='Element type')
    parser.add_argument('-I', '--init',
                        type=float,
                        required=False,
                        metavar='FLOAT',
                        dest='vol',
                        default=None,
                        help='Volume per atom')

    return parser


def parse_arguments():
    parser = generate_parser()
    args = vars(parser.parse_args())
    # parser.print_usage()
    # sys.exit(0)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    