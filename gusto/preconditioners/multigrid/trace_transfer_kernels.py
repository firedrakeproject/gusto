import coffee.base as ast
import numpy
import ufl
import gem
import gem.impero_utils as impero_utils

from tsfc.finatinterface import create_element
from tsfc import fem
from tsfc.parameters import default_parameters
from tsfc.coffee import SCALAR_TYPE, generate as generate_coffee

from finat.point_set import PointSet

from gem.interpreter import evaluate

from pyop2 import op2
from pyop2.datatypes import IntType

from firedrake.functionspacedata import cached, entity_dofs_key


def coefficient_matrix(Vf):

    # Only goes to P1 on simplex faces
    mesh = Vf.ufl_domain()
    assert mesh.ufl_cell().is_simplex()

    assert Vf.ufl_element().value_shape() == ()

    face = {"tetrahedron": ufl.triangle,
            "triangle": ufl.interval}[mesh.ufl_cell().cellname()]
    P1 = create_element(ufl.FiniteElement("P", face, 1))

    trace = create_element(ufl.FiniteElement("DP", face, Vf.ufl_element().degree()))

    points = []
    for dual in trace._element.dual_basis():
        pt, = dual.get_point_dict().keys()
        points.append(pt)

    ps = PointSet(points)
    matrix = fem.fiat_to_ufl(P1.basis_evaluation(0, ps), 0)

    indices = P1.get_indices()
    expr = gem.Indexed(matrix, indices)

    expr = gem.ComponentTensor(expr, indices + ps.indices)
    expr, = evaluate([expr])
    return gem.Literal(expr.arr)


def transfer_kernel(Vf, galerkin=False, restrict=True):

    mesh = Vf.ufl_domain()
    key = ((restrict, galerkin) + entity_dofs_key(Vf.finat_element.entity_dofs()))
    cache = mesh._shared_data_cache["transfer_kernels"]

    try:
        return cache[key]
    except KeyError:
        pass

    parameters = default_parameters()
    if restrict:
        matrix = coefficient_matrix(Vf)
    else:
        matrix = gem.Literal(coefficient_matrix(Vf).array.T)

    if galerkin:
        return_variable = gem.Variable("A", matrix.shape)
        indices = tuple(gem.Index(extent=e) for e in matrix.shape)

        expr = gem.Indexed(matrix, indices)
        return_variable = gem.Indexed(return_variable, indices)
        expr, = impero_utils.preprocess_gem([expr])
        assignments = [(return_variable, expr)]
        impero_c = impero_utils.compile_gem(assignments, indices, remove_zeros=True)

        body = generate_coffee(impero_c, {}, parameters["precision"])
        args = [ast.Decl(SCALAR_TYPE, ast.Symbol("A", rank=matrix.shape))]
        kernel = op2.Kernel(ast.FunDecl("void", "restriction_matrix",
                                        args, body, pred=["static", "inline"]),
                            name="restriction_matrix")
    else:
        oshape, ishape = matrix.shape
        coefficient = gem.Variable("R", (ishape, ))
        return_variable = gem.Variable("A", (oshape, ))
        indices = tuple(gem.Index(extent=e) for e in matrix.shape)
        i, j = indices

        return_variable = gem.Indexed(return_variable, (i, ))

        expr = gem.IndexSum(gem.Product(gem.Indexed(matrix, (i, j)),
                                        gem.Indexed(coefficient, (j, ))), (j, ))

        expr, = impero_utils.preprocess_gem([expr])
        assignments = [(return_variable, expr)]
        impero_c = impero_utils.compile_gem(assignments, indices, remove_zeros=True)

        body = generate_coffee(impero_c, {}, parameters["precision"])
        args = [ast.Decl(SCALAR_TYPE, ast.Symbol("A", rank=(oshape, ))),
                ast.Decl(SCALAR_TYPE, ast.Symbol("R"),
                         pointers=[("restrict",)],
                         qualifiers=["const"])]

        kernel = op2.Kernel(ast.FunDecl("void", "restriction_operator",
                                        args, body, pred=["static", "inline"]),
                            name="restriction_operator")

    return cache.setdefault(key, kernel)


def restrict(fine, coarse):
    kernel = transfer_kernel(fine.function_space(), restrict=True)

    fine_map = facet_map(fine.function_space())
    coarse_map = facet_map(coarse.function_space())

    mesh = fine.ufl_domain()

    coarse.dat.zero()
    op2.par_loop(kernel, mesh.facet_set,
                 coarse.dat(op2.INC, coarse_map[op2.i[0]]),
                 fine.dat(op2.READ, fine_map[op2.i[0]]))

    return coarse


def prolong(coarse, fine):

    kernel = transfer_kernel(fine.function_space(), restrict=False)
    fine_map = facet_map(fine.function_space())
    coarse_map = facet_map(coarse.function_space())

    mesh = fine.ufl_domain()
    op2.par_loop(kernel, mesh.facet_set,
                 fine.dat(op2.INC, fine_map[op2.i[0]]),
                 coarse.dat(op2.READ, coarse_map[op2.i[0]]))


def facet_map(V):

    mesh = V.ufl_domain()
    if hasattr(mesh, "facet_set"):
        facet_set = mesh.facet_set
    else:
        intfacets = mesh.interior_facets
        extfacets = mesh.exterior_facets
        intset = intfacets.set
        extset = extfacets.set

        core_size = extset.core_size + intset.core_size
        size = extset.size + intset.size
        total_size = extset.total_size + intset.total_size

        facet_set = op2.Set((core_size, size, total_size), name="facets")
        mesh.facet_set = facet_set

    key = entity_dofs_key(V.finat_element.entity_dofs())

    return get_facet_map(mesh, key, V)


@cached
def get_facet_map(mesh, key, V):

    intfacets = mesh.interior_facets
    extfacets = mesh.exterior_facets

    intset = intfacets.set
    extset = extfacets.set

    indexing = V.finat_element.entity_closure_dofs()[mesh.topological_dimension() - 1]

    indices = numpy.zeros((len(indexing), len(indexing[0])), dtype=int)
    for i, vals in indexing.items():
        indices[i, :] = vals

    core_size = mesh.facet_set.core_size
    size = mesh.facet_set.size
    total_size = mesh.facet_set.total_size

    facetmap = numpy.full((total_size, indices.shape[1]), -1, dtype=IntType)

    intvalues = V.interior_facet_node_map().values_with_halo
    extvalues = V.exterior_facet_node_map().values_with_halo

    facetmap[:extset.core_size, :] = extvalues[numpy.arange(extset.core_size).reshape(-1, 1),
                                               indices[extfacets.local_facet_number[:extset.core_size].flat]]
    facetmap[extset.core_size:core_size, :] = intvalues[numpy.arange(intset.core_size).reshape(-1, 1),
                                                        indices[intfacets.local_facet_number[:intset.core_size, 0].flat]]

    facetmap[core_size:core_size+extset.size, :] = extvalues[numpy.arange(extset.core_size, extset.size).reshape(-1, 1),
                                                             indices[extfacets.local_facet_number[extset.core_size:extset.size].flat]]
    facetmap[core_size+extset.size:size, :] = intvalues[numpy.arange(intset.core_size, intset.size).reshape(-1, 1),
                                                        indices[intfacets.local_facet_number[intset.core_size:intset.size, 0].flat]]

    facetmap[size:size+extset.total_size, :] = extvalues[numpy.arange(extset.size, extset.total_size).reshape(-1, 1),
                                                         indices[extfacets.local_facet_number[extset.size:extset.total_size].flat]]
    facetmap[size+extset.size:total_size, :] = intvalues[numpy.arange(intset.size, intset.total_size).reshape(-1, 1),
                                                         indices[intfacets.local_facet_number[intset.size:intset.total_size, 0].flat]]

    return op2.Map(mesh.facet_set, V.node_set, arity=indices.shape[1], values=facetmap)
