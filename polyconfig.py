import argparse
import json
import numpy as np


class PolyConfig:

    def __init__(self, **kwargs):

        arguments = Parser().parse_args()

        if arguments.file is None:
            raise NotImplementedError
        else:
            with open(arguments.file, 'r') as f:
                file_args = json.load(f)

            self.ngrains = file_args['grains']
            self.lattice_vectors = np.array(file_args['lattice_vectors'])
            self.basis = {el: np.array(vec) for el, vec in file_args['basis'].items()}
            self.size = np.array(file_args['size'])
            self.out = file_args['output_filename']
            self.dimension = file_args['dimension']
        return

    def __repr__(self):
        print(str(self))
        return

    def __str__(self):
        ret = 'PolyConfig()\n'
        ret += f'\tGrains: {self.ngrains}\n'
        ret += f'\tDimension: {self.dimension}\n'
        ret += f'\tSize: {self.size}\n'
        ret += '\tBasis:\n'
        for el, vec in self.basis.items():
            ret += f'\t\t{el}:\t{vec}\n'
        ret += '\tLattice Vectors:\n'
        for vec in self.lattice_vectors:
            ret += f'\t\t{vec}\n'
        return ret


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
        self.add_argument('-f', '--file',
                          type=str,
                          required=False,
                          metavar='FILE',
                          dest='file',
                          default='default.json',
                          help='Config file')

        return
