
import firedrake as fd

from ufl.classes import MultiIndex, FixedIndex, Indexed

from gusto.complex_proxy.common import (Part, re, im, api_names,  # noqa:F401
                                        ComplexConstant,  # noqa: F401
                                        _build_oneform, _build_twoform)

__all__ = api_names


def FiniteElement(elem):
    """
    Return a UFL FiniteElement which proxies a complex version of the real UFL FiniteElement elem.

    The returned complex-valued element has as many components as the real-valued element, but each component has a 'real' and 'imaginary' part eg:
    Scalar real elements become 2-vector complex elements.
    n-vector real elements become 2xn-tensor complex elements
    (shape)-tensor real elements become (2,shape)-tensor complex elements

    :arg elem: the UFL FiniteElement to be proxied
    """
    if isinstance(elem, fd.TensorElement):
        shape = (2,) + elem._shape
        scalar_element = elem.sub_elements[0]
        return fd.TensorElement(scalar_element, shape=shape)

    elif isinstance(elem, fd.VectorElement):
        shape = (2, elem.num_sub_elements)
        scalar_element = elem.sub_elements[0]
        return fd.TensorElement(scalar_element, shape=shape)

    elif isinstance(elem, fd.MixedElement):  # recurse
        return fd.MixedElement([FiniteElement(e) for e in elem.sub_elements])

    else:
        return fd.VectorElement(elem, dim=2)


def compatible_ufl_elements(elemc, elemr):
    """
    Return whether the ufl element elemc is a complex proxy for real ufl element elemr

    :arg elemc: complex proxy ufl element
    :arg elemr: real ufl element
    """
    return elemc == FiniteElement(elemr)


def FunctionSpace(V):
    """
    Return a FunctionSpace which proxies a complex version of the real FunctionSpace V.

    The returned complex-valued Function space has as many components as the real-valued FunctionSpace, but each component has a 'real' and 'imaginary' part eg:
    Scalar components of the real-valued FunctionSpace become 2-vector components of the complex-valued space.
    n-vector components of the real-valued FunctionSpace become 2xn-tensor components of the complex-valued space.
    (shape)-tensor components of the real-valued FunctionSpace become (2,shape)-tensor components of the complex-valued FunctionSpace.

    :arg V: the real-valued FunctionSpace.
    """
    return fd.FunctionSpace(V.mesh(), FiniteElement(V.ufl_element()))


def DirichletBC(W, V, bc, function_arg=None):
    """
    Return a DirichletBC on the complex FunctionSpace W that is equivalent to the DirichletBC bc on the real FunctionSpace V that W was constructed from.

    :arg W: the complex FunctionSpace.
    :arg V: the real FunctionSpace that W was constructed from.
    :arg bc: a DirichletBC on the real FunctionSpace that W was constructed from.
    """
    if function_arg is None:
        function_arg = bc.function_arg

    sub_domain = bc.sub_domain

    if type(V.ufl_element()) is fd.MixedElement:
        idx = bc.function_space().index
        Ws = (W.sub(idx).sub(0), W.sub(idx).sub(1))
    else:
        Ws = (W.sub(0), W.sub(1))

    return tuple(fd.DirichletBC(Wsub, function_arg, sub_domain) for Wsub in Ws)


def split(u, i):
    """
    If u is a Coefficient or Argument in the complex FunctionSpace,
        returns a tuple with the Function components corresponding
        to the real or imaginary subelements, indexed appropriately.

    :arg u: a Coefficient or Argument in the complex FunctionSpace
    :arg i: Part.Real for real subelements, Part.Imag for imaginary elements
    """
    if not isinstance(i, Part):
        raise ValueError("i must be a Part enum")

    us = fd.split(u)

    ncomponents = len(u.function_space().subfunctions)

    if ncomponents == 1:
        return tuple((us[i],))

    def get_sub_element(cpt, i):
        part = us[cpt]
        idxs = fd.indices(len(part.ufl_shape) - 1)
        return fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(i), *idxs))), idxs)

    return tuple(get_sub_element(cpt, i) for cpt in range(ncomponents))


def subfunctions(u, i):
    """
    Return a tuple of the real or imaginary components of the complex Function u. Analogous to u.subfunctions.

    :arg u: a complex Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if type(u) is tuple:
        return tuple(v.sub(i) for v in u)

    elem = u.ufl_element()
    if isinstance(elem, fd.TensorElement):
        num_sub_real = elem.num_sub_elements()//2
        return tuple((u.sub(i*num_sub_real + j) for j in range(num_sub_real)))

    elif isinstance(elem, fd.VectorElement):
        return tuple((u.sub(i),))

    elif isinstance(elem, fd.MixedElement):
        return tuple((w for v in u.subfunctions for w in subfunctions(v, i)))

    else:
        raise ValueError("u must be a Function from a complex-proxy FunctionSpace")


def _get_part(u, vout, i):
    """
    Get the real or imaginary part of the complex Function u and copy it to real Function vout.

    :arg u: a complex Function.
    :arg vout: a real Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    usub = subfunctions(u, i)
    vsub = vout if type(vout) is tuple else vout.subfunctions

    for csub, rsub in zip(usub, vsub):
        rsub.assign(csub)

    return vout


def _set_part(u, vnew, i):
    """
    Set the real or imaginary part of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg vnew: a real Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    usub = subfunctions(u, i)
    vsub = vnew if type(vnew) is tuple else vnew.subfunctions

    for csub, rsub in zip(usub, vsub):
        csub.assign(rsub)


def get_real(u, vout, name=None):
    """
    Return a real Function equal to the real component of the complex Function u.

    :arg u: a complex Function.
    :arg vout: If a real Function then real component of u is placed here. NotImplementedError(If None then a new Function is returned.)
    :arg name: If vout is None, the name of the new Function. Ignored if vout is not none.
    """
    if vout is None:
        raise NotImplementedError("Inferring real FunctionSpace from complex FunctionSpace not implemented yet")
    return _get_part(u, vout, Part.Real)


def get_imag(u, vout, name=None):
    """
    Return a real Function equal to the imaginary component of the complex Function u.

    :arg u: a complex Function.
    :arg vout: If a real Function then the imaginary component of u is placed here. NotImplementedError(If None then a new Function is returned.)
    :arg name: If vout is None, the name of the new Function. Ignored if uout is not none.
    """
    if vout is None:
        raise NotImplementedError("Inferring real FunctionSpace from complex FunctionSpace not implemented yet")
    return _get_part(u, vout, Part.Imag)


def set_real(u, vnew):
    """
    Set the real component of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg vnew: A real Function.
    """
    _set_part(u, vnew, Part.Real)


def set_imag(u, vnew):
    """
    Set the imaginary component of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg vnew: A real Function.
    """
    _set_part(u, vnew, Part.Imag)


def LinearForm(W, z, f, return_z=False):
    """
    Return a Linear Form on the complex FunctionSpace W equal to a complex multiple of a linear Form on the real FunctionSpace.
    If z = zr + i*zi is a complex number, v = vr + i*vi is a complex TestFunction, we want to construct the Form:
    <zr*vr,f> + i<zi*vi,f>

    :arg W: the complex-proxy FunctionSpace.
    :arg z: a complex number.
    :arg f: a generator function for a linear Form on the real FunctionSpace, callable as f(*v) where v are TestFunctions on the real FunctionSpace.
    :arg return_z: If true, return Constants for the real/imaginary parts of z used in the LinearForm.
    """
    return _build_oneform(W, z, f, split, return_z)


def BilinearForm(W, z, A, return_z=False):
    """
    Return a bilinear Form on the complex FunctionSpace W equal to a complex multiple of a bilinear Form on the real FunctionSpace.
    If z = zr + i*zi is a complex number, u = ur + i*ui is a complex TrialFunction, and b = br + i*bi is a complex linear Form, we want to construct a Form such that (zA)u=b

    (zA)u = (zr*A + i*zi*A)(ur + i*ui)
          = (zr*A*ur - zi*A*ui) + i*(zr*A*ui + zi*A*ur)

          = | zr*A   -zi*A | | ur | = | br |
            |              | |    |   |    |
            | zi*A    zr*A | | ui | = | bi |

    :arg W: the complex-proxy FunctionSpace
    :arg z: a complex number.
    :arg A: a generator function for a bilinear Form on the real FunctionSpace, callable as A(*u, *v) where u and v are TrialFunctions and TestFunctions on the real FunctionSpace.
    :arg return_z: If true, return Constants for the real/imaginary parts of z used in the BilinearForm.
    """
    return _build_twoform(W, z, A, fd.TrialFunction(W), split, return_z)


def derivative(z, F, u, return_z=False):
    """
    Return a bilinear Form equivalent to z*J where z is a complex number, J = dF/dw, F is a nonlinear Form on the real-valued space, and w is a Function in the real-valued space. The real and imaginary components of the complex Function u most both be equal to w for this operation to be valid.

    If z = zr + i*zi is a complex number, x = xr + i*xi is a complex Function, b = br + i*bi is a complex linear Form, J is the bilinear Form dF/dw, we want to construct a Form such that (zJ)x=b

    (zJ)x = (zr*J + i*zi*J)(xr + i*xi)
          = (zr*J*xr - zi*J*xi) + i*(zr*A*xi + zi*A*xr)

          = | zr*J   -zi*J | | xr | = | br |
            |              | |    |   |    |
            | zi*J    zr*J | | xi | = | bi |

    :arg z: a complex number.
    :arg F: a generator function for a nonlinear Form on the real FunctionSpace, callable as F(*u, *v) where u and v are Functions and TestFunctions on the real FunctionSpace.
    :arg u: the Function to differentiate F with respect to
    :arg return_z: If true, return Constants for the real/imaginary parts of z used in the BilinearForm.
    """
    def A(*args):
        return fd.derivative(F(*args), u)

    return _build_twoform(u.function_space(), z, A, u, split, return_z)
