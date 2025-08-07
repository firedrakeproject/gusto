import pytest


def make_dirname(test_name):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        return f'pytest_{test_name}_parallel'
    else:
        return f'pytest_{test_name}'


def test_linear_williamson_2_siqn():
    from linear_williamson_2 import linear_williamson_2
    test_name = 'linear_williamson_2'
    linear_williamson_2(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        siqn=True,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_linear_williamson_2_siqn_parallel():
    test_linear_williamson_2_siqn()


def test_linear_williamson_2_trap():
    from linear_williamson_2 import linear_williamson_2
    test_name = 'linear_williamson_2'
    linear_williamson_2(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        siqn=False,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_linear_williamson_2_trap_parallel():
    test_linear_williamson_2_trap()


def test_moist_convective_williamson_2():
    from moist_convective_williamson_2 import moist_convect_williamson_2
    test_name = 'moist_convective_williamson_2'
    moist_convect_williamson_2(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_moist_convective_williamson_2_parallel():
    test_moist_convective_williamson_2()


def test_moist_thermal_williamson_5():
    from moist_thermal_williamson_5 import moist_thermal_williamson_5
    test_name = 'moist_thermal_williamson_5'
    moist_thermal_williamson_5(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_moist_thermal_williamson_5_parallel():
    test_moist_thermal_williamson_5()


def test_shallow_water_1d_wave():
    from shallow_water_1d_wave import shallow_water_1d_wave
    test_name = 'shallow_water_1d_wave'
    shallow_water_1d_wave(
        ncells=20,
        dt=1.0e-4,
        tmax=1.0e-3,
        dumpfreq=2,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_shallow_water_1d_wave_parallel():
    test_shallow_water_1d_wave()


def test_thermal_williamson_2():
    from thermal_williamson_2 import thermal_williamson_2
    test_name = 'thermal_williamson_2'
    thermal_williamson_2(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_thermal_williamson_2_parallel():
    test_thermal_williamson_2()


def test_williamson_2():
    from williamson_2 import williamson_2
    test_name = 'williamson_2'
    williamson_2(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_williamson_2_parallel():
    test_williamson_2()


def test_williamson_5():
    from williamson_5 import williamson_5
    test_name = 'williamson_5'
    williamson_5(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_williamson_5_parallel():
    test_williamson_5()


def test_linear_thermal_galewsky_jet():
    from linear_thermal_galewsky_jet import linear_thermal_galewsky_jet
    test_name = 'linear_thermal_galewsky_jet'
    linear_thermal_galewsky_jet(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_linear_thermal_galewsky_jet_parallel():
    test_linear_thermal_galewsky_jet()


def test_moist_thermal_eqiuvb_gw():
    from moist_thermal_equivb_gw import moist_thermal_gw
    test_name = 'moist_thermal_gw'
    moist_thermal_gw(
        ncells_per_edge=4,
        dt=1800.,
        tmax=18000.,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def moist_thermal_eqiuvb_gw_parallel():
    test_moist_thermal_eqiuvb_gw()
