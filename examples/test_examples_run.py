# import pytest
from os.path import abspath, dirname  # basename
# import subprocess
import glob
# import sys
# import os


examples_dir = abspath(dirname(__file__))
example_files = glob.glob("%s/*/test*.py" % examples_dir)


# @pytest.fixture(params=glob.glob("%s/*/test*.py" % examples_dir),
#                 ids=lambda x: basename(x))
# def example_file(request):
#     return abspath(request.param)


# def test_example_runs(example_file, tmpdir, monkeypatch):
#     # This ensures that the test writes output in a temporary
#     # directory, rather than where pytest was run from.
#     monkeypatch.chdir(tmpdir)
#     subprocess.run(
#         [sys.executable, example_file, "--serial"],
#         check=True,
#         env=os.environ | {"PYOP2_CFLAGS": "-O0"}
#     )


# def test_example_runs_parallel(example_file, tmpdir, monkeypatch):
#     # This ensures that the test writes output in a temporary
#     # directory, rather than where pytest was run from.
#     monkeypatch.chdir(tmpdir)
#     subprocess.run(
#         ["mpiexec", "-n", "4", sys.executable, example_file, "--parallel"],
#         check=True,
#         env=os.environ | {"PYOP2_CFLAGS": "-O0"}
#     )
