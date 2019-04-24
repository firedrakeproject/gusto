pipeline {
    agent {
      docker {
        image 'firedrakeproject/firedrake-vanilla:latest'
        label 'firedrakeproject'
        args '-v /var/run/docker.sock:/var/run/docker.sock'
        alwaysPull true
      }
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
        stage('Install Gusto') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
'''
                }
            }
        }
        stage('Test') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
python $(which firedrake-clean)
python -m pytest -n 12 -v tests
'''
                }
            }
        }
        stage('Lint') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
make lint
'''
                }
            }
        }
    }
}
