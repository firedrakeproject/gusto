import pytest


def make_dirname(test_name):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        return f'pytest_{test_name}_parallel'
    else:
        return f'pytest_{test_name}'


def test_dcmip_3_1_meanflow():
    from dcmip_3_1_meanflow_quads import dcmip_3_1_meanflow
    test_name = 'dcmip_3_1_meanflow'
    dcmip_3_1_meanflow(
        ncells_per_edge=4,
        nlayers=4,
        dt=100,
        tmax=10000,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_dcmip_3_1_meanflow_parallel():
    test_dcmip_3_1_meanflow()


def test_dry_bryan_fritsch():
    from dry_bryan_fritsch import dry_bryan_fritsch
    test_name = 'dry_bryan_fritsch'
    dry_bryan_fritsch(
        ncolumns=20,
        nlayers=20,
        dt=2.0,
        tmax=20.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=4)
def test_dry_bryan_fritsch_parallel():
    test_dry_bryan_fritsch()


# Hydrostatic equations not currently working
@pytest.mark.xfail
def test_mountain_hydrostatic():
    from mountain_hydrostatic import mountain_hydrostatic
    test_name = 'mountain_hydrostatic'
    mountain_hydrostatic(
        ncolumns=20,
        nlayers=10,
        dt=5.0,
        tmax=50.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


# Hydrostatic equations not currently working
@pytest.mark.xfail
@pytest.mark.parallel(nprocs=4)
def test_mountain_hydrostatic_parallel():
    test_mountain_hydrostatic()


# Hydrostatic equations not currently working
@pytest.mark.xfail
def test_skamarock_klemp_hydrostatic():
    from skamarock_klemp_hydrostatic import skamarock_klemp_hydrostatic
    test_name = 'skamarock_klemp_hydrostatic'
    skamarock_klemp_hydrostatic(
        ncolumns=30,
        nlayers=5,
        dt=6.0,
        tmax=60.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


# Hydrostatic equations not currently working
@pytest.mark.xfail
@pytest.mark.parallel(nprocs=2)
def test_skamarock_klemp_hydrostatic_parallel():
    test_skamarock_klemp_hydrostatic()


def test_skamarock_klemp_nonhydrostatic():
    from skamarock_klemp_nonhydrostatic import skamarock_klemp_nonhydrostatic
    test_name = 'skamarock_klemp_nonhydrostatic'
    skamarock_klemp_nonhydrostatic(
        ncolumns=30,
        nlayers=5,
        dt=6.0,
        tmax=60.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_skamarock_klemp_nonhydrostatic_parallel():
    test_skamarock_klemp_nonhydrostatic()


def test_straka_bubble():
    from straka_bubble import straka_bubble
    test_name = 'straka_bubble'
    straka_bubble(
        nlayers=6,
        dt=4.0,
        tmax=40.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=3)
def test_straka_bubble_parallel():
    test_straka_bubble()


def test_unsaturated_bubble():
    from unsaturated_bubble import unsaturated_bubble
    test_name = 'unsaturated_bubble'
    unsaturated_bubble(
        ncolumns=20,
        nlayers=20,
        dt=1.0,
        tmax=10.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_unsaturated_bubble_parallel():
    test_unsaturated_bubble()