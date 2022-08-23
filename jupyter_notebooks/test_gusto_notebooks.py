#An automated script to test the gusto notebooks aren't broken

import pytest
import os
import subprocess
import glob

cwd = os.path.abspath(os.path.dirname(__file__))

@pytest.fixture(params=glob.glob(os.path.join(cwd, "*.ipynb")),
                ids=lambda x: os.path.basename(x))
def ipynb_file(request):
    return os.path.abspath(request.param)

def test_notebook_runs(ipynb_file, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    pytest = os.path.join(os.environ.get("VIRTUAL_ENV"), "bin", "pytest")
    subprocess.check_call([pytest, "--nbval-lax", ipynb_file])
    
print('yo')