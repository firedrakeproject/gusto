"""Objects describe chemical conversion and reaction processes."""

from firedrake import dx, split, Function
from firedrake.fml import subject
from gusto.core.labels import prognostic
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
                 species1_name='X', species2_name='X2'):
        """
        Args:
            equation (:class: 'PrognosticEquationSet'): the model's equation
            k1(float, optional): Reaction rate for species 1 (X). Defaults to a
                constant 1 over the domain.
            k2(float, optional): Reaction rate for species 2 (X2). Defaults to a
                constant 1 over the domain.
            species1_name(str, optional): Name of the first interacting species.
                Defaults to 'X'.
            species2_name(str, optional): Name of the second interacting
                species. Defaults to 'X2'.
        """

        label_name = 'terminator_toy'
        super().__init__(equation, label_name, parameters=None)

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

        test_1 = equation.tests[self.species1_idx]
        test_2 = equation.tests[self.species2_idx]

        Kx = k1*species2 - k2*(species1**2)

        source1_expr = test_1 * 2*Kx * dx
        source2_expr = test_2 * -Kx * dx

        equation.residual -= self.label(subject(prognostic(source1_expr, 'X'), self.Xq), self.evaluate)
        equation.residual -= self.label(subject(prognostic(source2_expr, 'X2'), self.Xq), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source/sink for the coalescence process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        pass
