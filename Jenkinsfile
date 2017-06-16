pipeline {
    agent {
        label 'linux'
    }
    environment {
        PATH = "/usr/local/bin:/usr/bin:/bin"
        CC = "mpicc"
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
                        sh 'pip2 install virtualenv'
                        sh 'curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install'
                        sh 'python2 ./firedrake-install --disable-ssh --minimal-petsc'
                        sh '$HOME/.local/bin/virtualenv --relocatable firedrake'
                    }
                }
            }
        }
        stage('Install Gusto') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
pip install -e .
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
firedrake-clean
py.test -v tests -n 4
'''
                }
            }
        }
    }
}
