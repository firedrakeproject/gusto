
import firedrake as fd

from gusto.complex_proxy.common import (Part, re, im, api_names,  # noqa:F401
                                        ComplexConstant,  # noqa: F401
                                        _flatten_tree,
                                        _build_oneform, _build_twoform)
__all__ = api_names


def FiniteElement(elem):
    """
    Return a UFL FiniteElement which proxies a complex version of the real-valued UFL FiniteElement elem.

    The returned complex-valued element has twice as many components as the real-valued element, with
    each component of the real-valued element having a corresponding 'real' and 'imaginary' part eg:
    Non-mixed real elements become 2-component MixedElements.
    Mixed real elements become MixedElements with 2*len(elem.num_sub_elements()) components.
    Nested MixedElements are flattened before being proxied.

    :arg elem: the UFL FiniteElement to be proxied
    """
    flat_elems = _flatten_tree(elem,
                               is_leaf=lambda e: type(e) is not fd.MixedElement,
                               get_children=lambda e: e.sub_elements)

    return fd.MixedElement([e for ee in zip(flat_elems, flat_elems) for e in ee])


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

    The returned complex-valued FunctionSpace has twice as many components as the real-valued element, with
    each component of the real-valued FunctionSpace having a corresponding 'real' and 'imaginary' part eg:
    Non-mixed real FunctionSpaces become 2-component MixedFunctionSpaces.
    Mixed real FunctionSpaces become MixedFunctionSpaces with 2*len(V.ufl_element().num_sub_elements()) components.
    Function spaces with nested MixedElements are flattened before being proxied.

    :arg V: the real-valued FunctionSpace.
    """
    return fd.FunctionSpace(V.mesh().unique(), FiniteElement(V.ufl_element()))


def DirichletBC(W, V, bc, function_arg=None):
    """
    Return a DirichletBC on the complex FunctionSpace W that is equivalent to the DirichletBC bc on the real FunctionSpace V that W was constructed from.

    :arg W: the complex FunctionSpace.
    :arg V: the real FunctionSpace that W was constructed from.
    :arg bc: a DirichletBC on the real FunctionSpace that W was constructed from.
    """
    if type(V.ufl_element()) is fd.MixedElement:
        off = 2*bc.function_space().index
    else:
        off = 0

    if function_arg is None:
        function_arg = bc.function_arg

    return tuple((fd.DirichletBC(W.sub(off+i), function_arg, bc.sub_domain)
                  for i in range(2)))


def _component_elements(us, i):
    """
    Return a tuple of the real or imaginary components of the iterable us

    :arg us: an iterable having the same number of elements as the complex FunctionSpace
                i.e. twice the number of components as the real FunctionSpace.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if not isinstance(i, Part):
        raise TypeError("i must be a Part enum")
    return tuple(us[i::2])


def split(u, i):
    """
    If u is a Coefficient or Argument in the complex FunctionSpace,
        returns a tuple with the Function components corresponding
        to the real or imaginary subelements, indexed appropriately.
        Analogous to firedrake.split(u)

    :arg u: a Coefficient or Argument in the complex FunctionSpace
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    return _component_elements(fd.split(u), i)


def subfunctions(u, i):
    """
    Return a tuple of the real or imaginary components of the complex Function u. Analogous to u.subfunctions.

    :arg u: a complex Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    usub = u if type(u) is tuple else u.subfunctions
    return _component_elements(usub, i)


def _get_part(u, vout, i):
    """
    Copy the real or imaginary part of the complex Function u into the real-valued Function vout.

    :arg u: a complex Function.
    :arg vout: a real-valued Function.
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


def get_real(u, vout):
    """
    Copy the real component of the complex Function u into the real-valued Function vout

    :arg u: a complex Function.
    :arg vout: A real-valued Function that real component of u is copied into.
    """
    return _get_part(u, vout, Part.Real)


def get_imag(u, vout, name=None):
    """
    Copy the imaginary component of the complex Function u into the real-valued Function vout

    :arg u: a complex Function.
    :arg vout: A real-valued Function that imaginary component of u is copied into.
    """
    return _get_part(u, vout, Part.Imag)


def set_real(u, vnew):
    """
    Copy the real-valued Function vnew into the real part of the complex Function u.

    :arg u: a complex Function.
    :arg vnew: A real-value Function.
    """
    _set_part(u, vnew, Part.Real)


def set_imag(u, vnew):
    """
    Copy the real-valued Function vnew into the imaginary part of the complex Function u.

    :arg u: a complex Function.
    :arg vnew: A real-value Function.
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
