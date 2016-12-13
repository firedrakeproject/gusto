from sympy import *

x   = Symbol('x')
z   = Symbol('z')
H   = Symbol('H')
Bu  = Symbol('Bu')
pi  = Symbol('pi')
L   = Symbol('L')
a   = Symbol('a')
Nsq = Symbol('Nsq')

def Z(z):
    return Bu*((z/H)-0.5)

def coth(x):
    return cosh(x)/sinh(x)

def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))

def template_target_strings():

    template = a*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*x/L)-n()*Bu*cosh(Z(z))*sin(pi*x/L))

    template_s = printing.ccode(template).replace('x','(x[0]-L)')
    template_s = template_s.replace('z','x[2]')

    return template_s
