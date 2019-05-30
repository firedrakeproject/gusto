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
sudo -u firedrake /bin/bash << Here
whoami
cd /home/firedrake
. /home/firedrake/firedrake/bin/activate
firedrake-update --install gusto || (cat firedrake-update.log && /bin/false)
chmod a+rwx /home/firedrake/firedrake/lib/python*/site-packages
chmod a+rwx /home/firedrake/firedrake/lib/python*/site-packages/easy-install.pth
chmod a+rwx /home/firedrake/firedrake/bin
chmod -R a+rwx /home/firedrake/firedrake/.cache
firedrake-status
Here
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
