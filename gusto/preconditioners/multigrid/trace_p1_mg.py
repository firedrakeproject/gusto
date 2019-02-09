import firedrake
# import ufl
import trace_transfer_kernels as tk

from firedrake.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
# from firedrake.utils import cached_property
# from pyop2.profiling import timed_region, timed_function

from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
# from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.mg import inject, prolong, restrict
from firedrake.mg.utils import get_level
from firedrake.mg.ufl_utils import coarsen as symbolic_coarsen
from firedrake.variational_solver import LinearVariationalProblem
from firedrake.petsc import PETSc

from functools import singledispatch, update_wrapper


def methoddispatch(func):

    dispatcher = singledispatch(func)

    def wrapper(*args, **kwargs):
        return dispatcher.dispatch(args[1].__class__)(*args, **kwargs)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


__all__ = ['TraceP1MGPC']


class TraceP1MGPC(PCBase):

    def initialize(self, pc):

        from firedrake import solving_utils

        prefix = pc.getOptionsPrefix() + "trace_p1_mg_"
        _, P = pc.getOperators()

        ctx = P.getPythonContext()
        if not isinstance(ctx, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        # Get the trace space from the Python context
        T = ctx.a.arguments()[0].function_space()

        # Functions for the residual and solution of the trace system
        self.residual = Function(T)
        self.solution = Function(T)

        # Linear MG doesn't need RHS, supply zero instead
        lvp = LinearVariationalProblem(a=ctx.a, L=0, u=self.solution)
        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")
        pmat_type = PETSc.Options().getString(prefix + "pmat_type", "aij")
        appctx = ctx.appctx
        new_ctx = solving_utils._SNESContext(lvp,
                                             mat_type=mat_type,
                                             pmat_type=pmat_type,
                                             appctx=appctx,
                                             options_prefix=prefix)
        self._ctx = new_ctx

        # Set up transfer operators
        transfer_operators = (
            firedrake.dmhooks.transfer_operators(
                T,
                restrict=self.my_restrict,
                inject=self.my_inject),
            firedrake.dmhooks.transfer_operators(FunctionSpace(T.mesh(), "P", 1),
                                                 prolong=self.my_prolong)
        )
        self._transfer_ops = transfer_operators

    def update(self, pc):
        raise NotImplementedError

    def apply(self, pc, x, y):
        raise NotImplementedError

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def view(self, pc, viewer=None):
        raise NotImplementedError

    # Coarsening strategies for our custom multigrid procedure
    @methoddispatch
    def my_coarsen(self, expr, callback, coefficient_mapping=None):
        return symbolic_coarsen(expr,
                                callback,
                                coefficient_mapping=coefficient_mapping)

    @my_coarsen.register(firedrake.functionspaceimpl.FunctionSpace)
    @my_coarsen.register(firedrake.functionspaceimpl.WithGeometry)
    def _coarsen_fs(self, V, callback, coefficient_mapping=None):
        hierarchy, level = get_level(V.ufl_domain())
        if level == len(hierarchy) - 1:
            mesh = callback(V.ufl_domain(), callback)
            return FunctionSpace(mesh, "CG", 1)
        else:
            return symbolic_coarsen(V,
                                    callback,
                                    coefficient_mapping=coefficient_mapping)

    # Transfer operators for the custom multigrid procedure
    @staticmethod
    def my_restrict(fine, coarse):
        hierarchy, level = get_level(fine.ufl_domain())
        if level == len(hierarchy) - 1:
            tk.restrict(fine, coarse)
        else:
            restrict(fine, coarse)

    @staticmethod
    def my_inject(fine, coarse):
        hierarchy, level = get_level(fine.ufl_domain())
        if level == len(hierarchy) - 1:
            # TODO: implement injection
            pass
        else:
            inject(fine, coarse)

    @staticmethod
    def my_prolong(coarse, fine):
        hierarchy, level = get_level(fine.ufl_domain())
        if level == len(hierarchy) - 1:
            tk.prolong(coarse, fine)
        else:
            prolong(fine, coarse)
