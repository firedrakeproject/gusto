pipeline {
    agent {
        label 'linux'
    }
    environment {
        PATH = "/usr/local/bin:/usr/bin:/bin"
        CC = "mpicc"
        PYTHONHASHSEED="1243123"
    }
    stages {
        stage('Clean') {
            steps {
                dir('build') {
                    deleteDir()
                }
            }
        }
        stage('Install Firedrake') {
            steps {
                sh 'mkdir build'
                dir('build') {
                    timestamps {
                        sh 'curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install'
                        sh 'python3 ./firedrake-install --disable-ssh --minimal-petsc'
                    }
                }
            }
        }
        stage('Install Gusto') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
python -m pip install -e .
'''
                }
            }
        }
        stage('Lint') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
make lint
'''
                }
            }
        }
        stage('Test') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
export PYOP2_CACHE_DIR=${VIRTUAL_ENV}/pyop2_cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=${VIRTUAL_ENV}/tsfc_cache
python $(which firedrake-clean)
python -m pytest -n 4 -v tests
'''
                }
            }
        }
    }
}
