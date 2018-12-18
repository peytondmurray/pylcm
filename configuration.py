import argparse
import numpy as  np


class Config:

    def __init__(self):

        arguments = Parser().parse_args()

        self.volume = arguments.volume
        self.grains = arguments.grains
        self.a = arguments.a
        self.b = arguments.b
        self.c = arguments.c
        self.type = arguments.type
        self.out = arguments.out
        self.input = arguments.input
        self.element = arguments.element
        self.init = arguments.init
        self.dimension = arguments.dimension

        return


class Parser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # parser = argparse.ArgumentParser(description='Generate polycrystal files for input into LAMMPs.')
        self.add_argument('-v', '--volume',
                          type=float,
                          required=False,
                          metavar='VALUE',
                          dest='volume',
                          default=np.nan,
                          help='Volume per atom')
        self.add_argument('-g', '--grains',
                          type=int,
                          required=False,
                          metavar='INT',
                          dest='grains',
                          default=1,
                          help='Number of grains')
        self.add_argument('-a' '--width',
                          type=float,
                          required=False,
                          metavar='VALUE',
                          dest='a',
                          default=10,
                          help='Grain width [nm]')
        self.add_argument('-b', '--height',
                          type=float,
                          required=False,
                          metavar='VALUE',
                          dest='b',
                          default=10,
                          help='Grain height [nm]')
        self.add_argument('-c', '--length',
                          type=float,
                          required=False,
                          metavar='VALUE',
                          dest='c',
                          default=10,
                          help='Grain length [nm]')
        self.add_argument('-t', '--type',
                          type=str,
                          required=False,
                          metavar='STRUCTURE',
                          dest='type',
                          default='fcc',
                          help='Lattice structure')
        self.add_argument('-o', '--out',
                          type=str,
                          required=False,
                          metavar='FILETYPE',
                          dest='out',
                          default='lmp',
                          help='Output filetype')
        self.add_argument('-i', '--input',
                          required=False,
                          metavar='VALUE',
                          dest='input',
                          default=False,
                          help='Input grain parameters file')
        self.add_argument('-e', '--element',
                          type=str,
                          required=False,
                          metavar='ELEMENT',
                          dest='element',
                          default='Xe',
                          help='Element type')
        self.add_argument('-I', '--init',
                          type=float,
                          required=False,
                          metavar='FLOAT',
                          dest='init',
                          default=None,
                          help='Init')
        self.add_argument('-d', '--dimension',
                          type=int,
                          required=False,
                          metavar='VALUE',
                          dest='dimension',
                          default=3,
                          help='Dimension')

        return