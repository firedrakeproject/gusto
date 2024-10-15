import pytest


def make_dirname(test_name):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        return f'pytest_{test_name}_parallel'
    else:
        return f'pytest_{test_name}'


def test_skamarock_klemp_compressible_bouss():
    from skamarock_klemp_compressible import skamarock_klemp_compressible_bouss
    test_name = 'skamarock_klemp_compressible_bouss'
    skamarock_klemp_compressible_bouss(
        ncolumns=30,
        nlayers=5,
        dt=6.0,
        tmax=60.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_skamarock_klemp_compressible_bouss_parallel():
    test_skamarock_klemp_compressible_bouss()


def test_skamarock_klemp_incompressible_bouss():
    from skamarock_klemp_incompressible import skamarock_klemp_incompressible_bouss
    test_name = 'skamarock_klemp_incompressible_bouss'
    skamarock_klemp_incompressible_bouss(
        ncolumns=30,
        nlayers=5,
        dt=60.0,
        tmax=12.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_skamarock_klemp_incompressible_bouss_parallel():
    test_skamarock_klemp_incompressible_bouss()


def test_skamarock_klemp_linear_bouss():
    from skamarock_klemp_linear import skamarock_klemp_linear_bouss
    test_name = 'skamarock_klemp_linear_bouss'
    skamarock_klemp_linear_bouss(
        ncolumns=30,
        nlayers=5,
        dt=60.0,
        tmax=12.0,
        dumpfreq=10,
        dirname=make_dirname(test_name)
    )


@pytest.mark.parallel(nprocs=2)
def test_skamarock_klemp_linear_bouss_parallel():
    test_skamarock_klemp_linear_bouss()
