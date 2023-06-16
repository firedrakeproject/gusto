import numpy

original_constants = True

class REXIConstants(object):   # REXIParameters(object)

    if original_constants:
        """
        mu and a coefficients from
        "A high-order time-parallel scheme for solving wave propagation problems
        via the direct construction of an approximate time-evolution operator",
        Haut et.al.
        """

        mu = -4.315321510875024 + 1j*0
        L = 11
        a = [
            -1.0845749544592896e-7 + 1j*2.77075431662228e-8,
            1.858753344202957e-8 + 1j*-9.105375434750162e-7,
            3.6743713227243024e-6 + 1j*7.073284346322969e-7,
            -2.7990058083347696e-6 + 1j*0.0000112564827639346,
            0.000014918577548849352 + 1j*-0.0000316278486761932,
            -0.0010751767283285608 + 1j*-0.00047282220513073084,
            0.003816465653840016 + 1j*0.017839810396560574,
            0.12124105653274578 + 1j*-0.12327042473830248,
            -0.9774980792734348 + 1j*-0.1877130220537587,
            1.3432866123333178 + 1j*3.2034715228495942,
            4.072408546157305 + 1j*-6.123755543580666,
            -9.442699917778205 + 1j*0.,
            4.072408620272648 + 1j*6.123755841848161,
            1.3432860877712938 + 1j*-3.2034712658530275,
            -0.9774985292598916 + 1j*0.18771238018072134,
            0.1212417070363373 + 1j*0.12326987628935386,
            0.0038169724770333343 + 1j*-0.017839242222443888,
            -0.0010756025812659208 + 1j*0.0004731874917343858,
            0.000014713754789095218 + 1j*0.000031358475831136815,
            -2.659323898804944e-6 + 1j*-0.000011341571201752273,
            3.6970377676364553e-6 + 1j*-6.517457477594937e-7,
            3.883933649142257e-9 + 1j*9.128496023863376e-7,
            -1.0816457995911385e-7 + 1j*-2.954309729192276e-8
        ]
    else:
        """
        mu and a coefficients from "An accurate and time-parallel rational
        exponential integrator for hyperbolic and oscillatory PDEs",
        Caliari et.al. 2021
        """

        mu = -5.133333333333333 + 1j*0
        L = 12
        a = [
            -6.520430828919864e1 + 1j*0,
            4.261818064131437e1 + 1j*2.761406741120911e1,
            -9.801650304425239 + 1j*-2.189295463610722e1,
            -1.054225194693395 + 1j*6.791786454153551,
            7.950505668209775e-1 + 1j*-8.904997258367445e-1,
            -1.218558380859130e-1 + 1j*3.321241563407446e-2,
            7.365401806949337e-3 + 1j*2.212802103193251e-3,
            -2.801087265991056e-4 + 1j*-5.566945197754387e-4,
            1.254835436432561e-4 + 1j*-2.467200513365371e-4,
            2.295472292491263e-4 + 1j*-8.494118951459107e-5,
            1.858484460459430e-4 + 1j*9.242889460185034e-5,
            4.068056518449676e-5 + 1j*1.653479957565515e-4,
            -8.341508001647741e-5 + 1j*1.045331460447588e-4,
            -9.970528169841103e-5 + 1j*-5.856228484297677e-6,
            -3.499639858693093e-5 + 1j*-6.129059473910835e-5,
            2.295021920298455e-5 + 1j*-4.099832469456381e-5,
            2.931048772724314e-5 + 1j*1.708815129697846e-7,
            7.502088478301169e-6 + 1j*1.525082051744077e-5,
            -5.815291167450100e-6 + 1j*6.919604247338349e-6,
            -4.069948458364005e-6 + 1j*-1.440010113050771e-6,
            7.932524475429588e-8 + 1j*-1.794169428574330e-6,
            6.120984882186265e-7 + 1j*-1.131894636585849e-7,
            5.531365159161319e-8 + 1j*1.585749903175946e-7,
            -2.867805871375946e-8 + 1j*1.239499740327838e-8,
            -1.143081277095316e-9 + 1j*-2.763239274253499e-9
        ]

def b_coefficients(h, M):
    """
    Compute the b coefficients where
    b_m = h^2 exp(imh)
    """
    m = numpy.arange(-M, M+1)
    return numpy.exp(h*h)*numpy.exp(-1j*m*h)


def RexiCoefficients(rexi_parameters):
    """
    Compute the A_n and B_n coefficients in the REXI sum

    exp(ix) \approx sum_{n=0}^{P} B_n (ix + A_n)

    where P = 4N+1 if rexi_parameters.reduce_to_half is False else
    2N+1 if True.

    Returns 3 numpy arrays:
    alpha contains the A_n coefficients
    beta contains the B_n coefficients
    beta2 contains zeros if rexi_parameters.reduce_to_half is False else
    it contains the coefficients that multiply the conjugate terms

    :arg rexi_parameters: class containing the parameters (h, M and
    reduce_to_half) necessary to compute the coefficients

    """

    h = rexi_parameters.h
    M = rexi_parameters.M

    # get L, mu and the a coefficients
    constants = REXIConstants()
    L = constants.L
    mu = constants.mu
    a = constants.a

    # calculate the b coefficients
    b = b_coefficients(h, M)

    # allocate arrays for alpha, beta_re and beta_im
    N = M + L
    alpha = numpy.zeros((2*N+1,), dtype=numpy.complex128)
    beta_re = numpy.zeros((2*N+1,), dtype=numpy.complex128)
    beta_im = numpy.zeros((2*N+1,), dtype=numpy.complex128)

    # compute alpha, beta_re and beta_im
    for l in range(-L, L+1):
        for m in range(-M, M+1):
            n = l+m
            alpha[n+N] = h*(mu + 1j*n)
            beta_re[n+N] += b[m+M].real*h*a[l+L]
            beta_im[n+N] += b[m+M].imag*h*a[l+L]

    # calculate conj(beta_re) and conj(beta_im), used to define A_n and B_n
    beta_conj_re = numpy.conjugate(beta_re)
    beta_conj_im = numpy.conjugate(beta_im)

    if rexi_parameters.reduce_to_half:
        # If reducing the number of solvers to (nearly) half, as
        # described in notes.pdf, we only need alpha_n for n \in [N,
        # 2N]. We need to retain all the betas (due to the lack of
        # symmetry coming from the numerical calculation of the a
        # coefficients) but in order not to special case the Nth
        # solver (where the solution is real hence equal to its
        # conjugate) we must divide the Nth value of the betas by 2.
        alpha = alpha[N:]
        beta_re[N] /= 2.
        beta_im[N] /= 2.
        beta_conj_re[N] /= 2.
        beta_conj_im[N] /= 2.
        beta = numpy.concatenate(
            (beta_re[N:] + 1j*beta_im[N:],
             -beta_conj_re[N::-1] - 1j*beta_conj_im[N::-1])
        )/2
        # beta2 is the coefficient that multiplies the conjugate of
        # the solution
        beta2 = numpy.concatenate(
            (beta_re[N::-1] + 1j*beta_im[N::-1],
             -beta_conj_re[N:] - 1j*beta_conj_im[N:])
        )/2
    else:
        beta = numpy.concatenate(
            (beta_re + 1j*beta_im,
             -beta_conj_re[::-1] - 1j*beta_conj_im[::-1])
        )/2
        # when not reducing the number of solvers we return zero here
        # in order not to special case this option (i.e. we still
        # calculate the conjugate of the solution but then multiply it
        # by zero - you're doing (nearly) twice the amount of work necessary
        # anyway!)
        beta2 = numpy.zeros(len(beta))

    alpha = numpy.concatenate((alpha, -alpha))

    return alpha, beta, beta2
