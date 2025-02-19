"""
This provides an operator for perform a conservative projection.

The :class:`ConservativeProjector` provided in this module is an operator that
projects a field such as a mixing ratio from one function space to another,
weighted by a density field to ensure that mass is conserved by the projection.
"""

from firedrake import (Function, TestFunction, TrialFunction, lhs, rhs, inner,
                       dx, LinearVariationalProblem, LinearVariationalSolver,
                       Constant, assemble)
import ufl

__all__ = ["ConservativeProjector"]


class ConservativeProjector(object):
    """
    Projects a field such that mass is conserved.

    This object is designed for projecting fields such as mixing ratios of
    tracer species from one function space to another, but weighted by density
    such that mass is conserved by the projection.
    """

    def __init__(self, rho_source, rho_target, m_source, m_target,
                 subtract_mean=False):
        """
        Args:
            rho_source (:class:`Function`): the density to use for weighting the
                source mixing ratio field. Can also be a :class:`ufl.Expr`.
            rho_target (:class:`Function`): the density to use for weighting the
                target mixing ratio field. Can also be a :class:`ufl.Expr`.
            m_source (:class:`Function`): the source mixing ratio field. Can
                also be a :class:`ufl.Expr`.
            m_target (:class:`Function`): the target mixing ratio field to
                compute.
            subtract_mean (bool, optional): whether to solve the projection by
                subtracting the mean value of m for both sides. This is more
                expensive as it involves calculating the mean, but will ensure
                preservation of a constant when projecting to a continuous
                space. Default to False.

        Raises:
            RuntimeError: the geometric shape of the two rho fields must be equal.
            RuntimeError: the geometric shape of the two m fields must be equal.
        """

        self.subtract_mean = subtract_mean

        if not isinstance(rho_source, (ufl.core.expr.Expr, Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(rho_source))

        if not isinstance(rho_target, (ufl.core.expr.Expr, Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(rho_target))

        if not isinstance(m_source, (ufl.core.expr.Expr, Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(m_source))

        # Check shape values
        if m_source.ufl_shape != m_target.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (m_source.ufl_shape, m_target.ufl_shape))

        if rho_source.ufl_shape != rho_target.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (rho_source.ufl_shape, rho_target.ufl_shape))

        self.m_source = m_source
        self.m_target = m_target

        V = self.m_target.function_space()
        mesh = V.mesh()

        self.m_mean = Constant(0.0)
        self.volume = assemble(
            1*dx(domain=ufl.domain.extract_unique_domain(self.m_target))
            )

        test = TestFunction(V)
        m_trial = TrialFunction(V)
        eqn = (rho_source*inner(test, m_source - self.m_mean)*dx
               - rho_target*inner(test, m_trial - self.m_mean)*dx)
        problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), self.m_target)
        self.solver = LinearVariationalSolver(problem)

    def project(self):
        """Apply the projection."""

        # Compute mean value
        if self.subtract_mean:
            self.m_mean.assign(assemble(self.m_source*dx) / self.volume)

        # Solve projection
        self.solver.solve()

        return self.m_target
