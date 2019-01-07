import pytest
import polyconfig as pco
import polycrystal as pcr


def test_polyconfig_default():
    config = pco.PolyConfig(file='default.json')
    print(config)
    return


def test_polycrystal_default():
    config = pco.PolyConfig(file='default.json')
    crystal = pcr.Polycrystal(config)
    return


@pytest.mark.skip()
def cout_polycrystal_default():
    config = pco.PolyConfig(file='default.json')
    crystal = pcr.Polycrystal(config)
    crystal.initialize_grain_centers()
    for i in range(crystal.config.ngrains):
        crystal.generate_lattice(i)
    print(crystal)
    return


if __name__ == '__main__':
    # test_polyconfig_default()
    # test_polycrystal_default()
    cout_polycrystal_default()
