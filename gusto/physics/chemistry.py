"""Objects describe chemical conversion and reaction processes."""

from firedrake import dx, split, Function, sqrt, exp, Constant, conditional, max_value, min_value, assemble
from firedrake.__future__ import interpolate
from firedrake.fml import subject
from gusto.core.labels import prognostic, source_label
from gusto.core.logging import logger
from gusto.physics.physics_parametrisation import PhysicsParametrisation

__all__ = ["TerminatorToy"]


class TerminatorToy(PhysicsParametrisation):
    """
    Setup the Terminator Toy chemistry interaction
    as specified in 'The terminator toy chemistry test ...'
    Lauritzen et. al. (2014).

    The coupled equations for the two species are given by:

    D/Dt (X) = 2Kx
    D/Dt (X2) = -Kx

    where Kx = k1*X2 - k2*(X**2)
    """

    def __init__(self, equation, k1=1, k2=1,
                 species1_name='X_tracer', species2_name='X2_tracer',
                 analytical_formulation=False):
        """
        Args:
            equation (:class: 'PrognosticEquationSet'): the model's equation
            k1(float, optional): Reaction rate for species 1 (X). Defaults to a
                constant 1 over the domain.
            k2(float, optional): Reaction rate for species 2 (X2). Defaults to a
                constant 1 over the domain.
            species1_name(str, optional): Name of the first interacting species.
                Defaults to 'X_tracer'.
            species2_name(str, optional): Name of the second interacting
                species. Defaults to 'X2_tracer'.
            analytical_formulation (bool, optional): whether the scheme is already
                put into a Backwards Euler formulation (which allows this scheme
                to actually be used with a Forwards Euler or other explicit time
                discretisation). Otherwise, this is formulated more generally
                and can be used with any time stepper. Defaults to False.
        """

        label_name = 'terminator_toy'
        super().__init__(equation, label_name, parameters=None)

        self.analytical_formulation = analytical_formulation

        if species1_name not in equation.field_names:
            raise ValueError(f"Field {species1_name} does not exist in the equation set")
        if species2_name not in equation.field_names:
            raise ValueError(f"Field {species2_name} does not exist in the equation set")

        self.species1_idx = equation.field_names.index(species1_name)
        self.species2_idx = equation.field_names.index(species2_name)

        assert equation.function_space.sub(self.species1_idx) == equation.function_space.sub(self.species2_idx), \
            "The function spaces for the two species need to be the same"

        self.Xq = Function(equation.X.function_space())

        species1 = split(self.Xq)[self.species1_idx]
        species2 = split(self.Xq)[self.species2_idx]

        test1 = equation.tests[self.species1_idx]
        test2 = equation.tests[self.species2_idx]

        W = equation.function_space
        V_idxs = [self.species1_idx, self.species2_idx]

        self.dt = Constant(0.0)

        if analytical_formulation:
            # Implement this such at the chemistry forcing can never
            # make the mixing ratio fields become negative.
            # This uses the anaytical formulation given in
            # the DCMIP 2016 test case document.
            # This must be used with forwards Euler.
            X_T_0 = 4.e-6
            r = k1/(4.0*k2)
            d = sqrt(r**2 + 2.0*X_T_0*r)
            e = exp(-4.0*k2*d*self.dt)

            e1 = conditional(abs(d*k2*self.dt) > 1.e-16, (1.0-e)/(d*self.dt), 4.0*k2)

            source_expr = -e1 * (species1 - d + r) * (species1 + d + r) / (1.0 + e + self.dt * e1 * (species1 + r))

            # Ensure the increments don't make a negative mixing ratio
            source_expr = conditional(source_expr < 0.0,
                                      max_value(source_expr, -species1/self.dt),
                                      min_value(source_expr, 2.0*species2/self.dt))
            
            # Ensure the increments don't make a negative mixing ratio
            # AND X doesn't get larger than 4e-6,
            # AND X2 doesn't get larger than 2e-6
            #source_expr = conditional(source_expr < 0.0,
            #                          max_value(max_value(source_expr, -species1/self.dt), 2.0*(species2 - 2.e-6)/self.dt),
            #                          min_value(min_value(source_expr, 2.0*species2/self.dt), (4.e-6 - species1)/self.dt))

            source1_expr = source_expr
            source2_expr = -source_expr/2.0

            self.source = Function(W)
            self.source_expr = [split(self.source)[V_idx] for V_idx in V_idxs]
            self.source_int = [self.source.subfunctions[V_idx] for V_idx in V_idxs]

            self.source_interpolate = [interpolate(source1_expr, self.source_int[0]),
                                       interpolate(source2_expr, self.source_int[1])]

            equation.residual -= source_label(self.label(subject(prognostic(test1 * self.source_expr[0] * dx, species1_name), equation.X), self.evaluate))
            equation.residual -= source_label(self.label(subject(prognostic(test2 * self.source_expr[1] * dx, species2_name), equation.X), self.evaluate))

        else:
            Kx = k1*species2 - k2*(species1**2)

            source1_expr = 2*Kx
            source2_expr = -Kx

            source1_expr = test1 * source1_expr * dx
            source2_expr = test2 * source2_expr * dx

            equation.residual -= self.label(subject(prognostic(source1_expr, species1_name), self.Xq), self.evaluate)
            equation.residual -= self.label(subject(prognostic(source2_expr, species2_name), self.Xq), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the Terminator Toy interaction chemistry.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """

        self.dt.assign(dt)

        if self.analytical_formulation:
            self.Xq.assign(x_in)

            # Evaluate the source
            for interpolator, src in zip(self.source_interpolate, self.source_int):
                src.assign(assemble(interpolator))

            if x_out is not None:
                x_out.assign(self.source)

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        pass
