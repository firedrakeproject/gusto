"""
This script can be used to generate up-to-date KGOs (known good output files)
for Gusto integration tests. It runs each test case in a temporary directory
and copies the checkpoint file into the integration-tests/data directory.

If a new test is introduced, the corresponding function should be added to the
list of imports and the test_dict dictionary below.

Note that this script can take a while (~15 minutes) to run from a cold cache.
"""

from functools import partial
from os.path import join, abspath, dirname
import os
import tempfile
import shutil
import sys

# ---------------------------------------------------------------------------- #
# First, add integration-tests to python path so that we can import functions
# from tests for each equation
local_dir = dirname(abspath(__file__))  # Path to "data"
parent_dir = dirname(local_dir)         # Path to "integration-tests"
sys.path.append(parent_dir)             # Add "integration-tests" to python path
test_dir = 'kgo'                        # Name of directory to pass to tests

# ---------------------------------------------------------------------------- #
# Import KGO test cases
# ---------------------------------------------------------------------------- #
from equations.test_boussinesq import run_boussinesq
from equations.test_dry_compressible import run_dry_compressible
from equations.test_linear_sw_wave import run_linear_sw_wave
from equations.test_moist_compressible import run_moist_compressible
from equations.test_sw_fplane import run_sw_fplane
from model.test_simultaneous_SIQN import run_simult_SIQN

# ---------------------------------------------------------------------------- #
# List all test cases
# ---------------------------------------------------------------------------- #

# Dictionary of test cases, where:
# - the key is the name of the test case, and
# - the value is the test case function to be run
# NB: the keys should match the names of directories in the test routines,
# and the names of the saved checkpoint files
test_dict = {
    'boussinesq_compressible': partial(run_boussinesq, compressible=True),
    'boussinesq_incompressible': partial(run_boussinesq, compressible=False),
    'dry_compressible': run_dry_compressible,
    'linear_sw_wave': run_linear_sw_wave,
    'moist_compressible': run_moist_compressible,
    'sw_fplane': run_sw_fplane,
    'simult_SIQN_order0': partial(run_simult_SIQN, order=0),
    'simult_SIQN_order1': partial(run_simult_SIQN, order=1)
}

# ---------------------------------------------------------------------------- #
# Create a temporary directory
# ---------------------------------------------------------------------------- #
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Created temporary directory: {temp_dir}")

    # Store original directory and change to the temporary directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    successes = []
    failures = []

    # ------------------------------------------------------------------------ #
    # Loop through test cases
    # ------------------------------------------------------------------------ #
    for test_name, test_func in test_dict.items():
        print('='*60)
        print(f'Running {test_name}')
        print('='*60)

        try:
            test_func(test_dir)

            # ---------------------------------------------------------------- #
            # Copy checkpoint file to "data" directory
            # ---------------------------------------------------------------- #
            # Source file is e.g. in /tmp/tmp_dir/results/kgo/sw_fplane/chkpt.h5
            source_file = join(temp_dir, 'results', test_dir, test_name, 'chkpt.h5')

            # Destination file is e.g. integration-tests/data/sw_fplane_chkpt.h5
            dest_file = join(local_dir, f'{test_name}_chkpt.h5')

            # Copy the file from the temporary directory to the destination
            shutil.copy(source_file, dest_file)

            successes.append(test_name)

        except Exception as e:
            failures.append((test_name, e))

    # Change back to the original directory
    os.chdir(original_dir)

# ---------------------------------------------------------------------------- #
# Summarise output
# ---------------------------------------------------------------------------- #
print('='*60)
print('SUMMARY')
print('='*60)
for test_name in successes:
    print(f'{test_name} successfully copied over KGO')
print('='*60)
for test_name, e in failures:
    print(f'{test_name} failed with error: {e}')
print('='*60)
