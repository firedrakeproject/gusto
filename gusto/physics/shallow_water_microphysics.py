"""
Defines microphysics routines to be used with the moist shallow water equations.
"""

from firedrake import (
    conditional, Function, dx, min_value, max_value, FunctionSpace,
    assemble, split
)
from firedrake.__future__ import interpolate
from firedrake.fml import subject
from gusto.core.logging import logger
from gusto.physics.physics_parametrisation import PhysicsParametrisation
from gusto.core.labels import source_label
from types import FunctionType


__all__ = ["InstantRain", "SWSaturationAdjustment"]


class InstantRain(PhysicsParametrisation):
    """
    The process of converting vapour above the saturation curve to rain.

    A scheme to move vapour directly to rain. If convective feedback is true
    then this process feeds back directly on the height equation. If rain is
    accumulating then excess vapour is being tracked and stored as rain;
    otherwise converted vapour is not recorded. The process can happen over the
    timestep dt or over a specified time interval tau.
     """

    def __init__(self, equation, saturation_curve,
                 time_varying_saturation=False,
                 vapour_name="water_vapour", rain_name=None, gamma_r=1,
                 convective_feedback=False, beta1=None, tau=None,
                 parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            saturation_curve (:class:`ufl.Expr` or func): the curve above which
                excess moisture is converted to rain. Is either prescribed or
                dependent on a prognostic field.
            time_varying_saturation (bool, optional): set this to True if the
                saturation curve is changing in time. Defaults to False.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to "water_vapour".
            rain_name (str, optional): name of the rain variable. Defaults to
                None.
            gamma_r (float, optional): Fraction of vapour above the threshold
                which is converted to rain. Defaults to one, in which case all
                vapour above the threshold is converted.
            convective_feedback (bool, optional): True if the conversion of
                vapour affects the height equation. Defaults to False.
            beta1 (float, optional): Condensation proportionality constant,
                used if convection causes a response in the height equation.
                Defaults to None, but must be specified if convective_feedback
                is True.
            tau (float, optional): Timescale for condensation. Defaults to None,
                in which case the timestep dt is used.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of gas constants. Defaults to None, in which case the
                parameters are obtained from the equation.
        """

        self.explicit_only = True
        label_name = 'instant_rain'
        super().__init__(equation, label_name, parameters=parameters)

        self.convective_feedback = convective_feedback
        self.time_varying_saturation = time_varying_saturation

        # check for the correct fields
        assert vapour_name in equation.field_names, f"Field {vapour_name} does not exist in the equation set"
        self.Vv_idx = equation.field_names.index(vapour_name)

        if rain_name is not None:
            assert rain_name in equation.field_names, f"Field {rain_name} does not exist in the equation set "

        if self.convective_feedback:
            assert "D" in equation.field_names, "Depth field must exist for convective feedback"
            assert beta1 is not None, "If convective feedback is used, beta1 parameter must be specified"

        # obtain function space and functions; vapour needed for all cases
        W = equation.function_space
        Vv = W.sub(self.Vv_idx)
        test_v = equation.tests[self.Vv_idx]

        # depth needed if convective feedback
        if self.convective_feedback:
            self.VD_idx = equation.field_names.index("D")
            VD = W.sub(self.VD_idx)
            test_D = equation.tests[self.VD_idx]
            self.D = Function(VD)

        # the source function is the difference between the water vapour and
        # the saturation function
        self.water_v = Function(Vv)
        self.source = Function(W)
        self.source_expr = split(self.source)[self.Vv_idx]
        self.source_int = self.source.subfunctions[self.Vv_idx]

        R = FunctionSpace(equation.domain.mesh, "R", 0)
        # tau is the timescale for conversion (may or may not be the timestep)
        if tau is not None:
            self.set_tau_to_dt = False
            self.tau = Function(R).assign(tau)
        else:
            self.set_tau_to_dt = True
            self.tau = Function(R)
            logger.info("Timescale for rain conversion has been set to dt. If this is not the intention then provide a tau parameter as an argument to InstantRain.")

        if self.time_varying_saturation:
            if isinstance(saturation_curve, FunctionType):
                self.saturation_computation = saturation_curve
                self.saturation_curve = Function(Vv)
            else:
                raise NotImplementedError(
                    "If time_varying_saturation is True then saturation must be a Python function of a prognostic field.")
        else:
            assert not isinstance(saturation_curve, FunctionType), "If time_varying_saturation is not True then saturation cannot be a Python function."
            self.saturation_curve = saturation_curve

        # lose proportion of vapour above the saturation curve
        equation.residual += source_label(
            self.label(subject(test_v * self.source_expr * dx, self.source), self.evaluate)
        )

        # if rain is not none then the excess vapour is being tracked and is
        # added to rain
        if rain_name is not None:
            self.Vr_idx = equation.field_names.index(rain_name)
            test_r = equation.tests[self.Vr_idx]
            equation.residual -= source_label(
                self.label(subject(test_r * self.source_expr * dx, self.source), self.evaluate)
            )

        # if feeding back on the height adjust the height equation
        if convective_feedback:
            self.VD_idx = equation.field_names.index("D")
            equation.residual += source_label(
                self.label(subject(test_D * beta1 * self.source * dx, equation.X), self.evaluate)
            )

        # interpolator does the conversion of vapour to rain
        self.source_interpolate = interpolate(conditional(
            self.water_v > self.saturation_curve,
            (1/self.tau)*gamma_r*(self.water_v - self.saturation_curve),
            0), self.source_int.function_space())

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evalutes the source term generated by the physics.

        Computes the physics contributions (loss of vapour, accumulation of
        rain and loss of height due to convection) at each timestep.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        if self.convective_feedback:
            self.D.assign(x_in.subfunctions[self.VD_idx])
        if self.time_varying_saturation:
            self.saturation_curve.interpolate(self.saturation_computation(x_in))
        if self.set_tau_to_dt:
            self.tau.assign(dt)
        self.water_v.assign(x_in.subfunctions[self.Vv_idx])

        assemble(self.source_interpolate, tensor=self.source_int)

        if x_out is not None:
            x_out.assign(self.source)


class SWSaturationAdjustment(PhysicsParametrisation):
    """
    Represents the process of adjusting water vapour and cloud water according
    to a saturation function, via condensation and evaporation processes.

    This physics scheme follows that of Zerroukat and Allen (2015).

    """

    def __init__(self, equation, saturation_curve,
                 time_varying_saturation=False, vapour_name='water_vapour',
                 cloud_name='cloud_water', convective_feedback=False,
                 beta1=None, thermal_feedback=False, beta2=None, gamma_v=1,
                 time_varying_gamma_v=False, tau=None,
                 parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation
            saturation_curve (:class:`ufl.Expr` or func): the curve which
                dictates when phase changes occur. In a saturated atmosphere
                vapour above the saturation curve becomes cloud, and if the
                atmosphere is sub-saturated and there is cloud present cloud
                will become vapour until the saturation curve is reached. The
                saturation curve is either prescribed or dependent on a
                prognostic field.
            time_varying_saturation (bool, optional): set this to True if the
                saturation curve is changing in time. Defaults to False.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to 'water_vapour'.
            cloud_name (str, optional): name of the cloud variable. Defaults to
                'cloud_water'.
            convective_feedback (bool, optional): True if the conversion of
                vapour affects the height equation. Defaults to False.
            beta1 (float, optional): Condensation proportionality constant for
                height feedback, used if convection causes a response in the
                height equation. Defaults to None, but must be specified if
                convective_feedback is True.
            thermal_feedback (bool, optional): True if moist conversions
                affect the buoyancy equation. Defaults to False.
            beta2 (float, optional): Condensation proportionality constant
                for thermal feedback. Defaults to None, but must be specified
                if thermal_feedback is True. This is equivalent to the L_v
                parameter in Zerroukat and Allen (2015).
            gamma_v (ufl expression or :class: `function`): The proportion of
                moist species that is converted when a conversion between
                vapour and cloud is taking place. Defaults to one, in which
                case the full amount of species to bring vapour to the
                saturation curve will undergo a conversion. Converting only a
                fraction avoids a two-timestep oscillation between vapour and
                cloud when saturation is tempertature/height-dependent.
            time_varying_gamma_v (bool, optional): set this to True
                if the fraction of moist species converted changes in time
                (if gamma_v is temperature/height-dependent).
            tau (float, optional): Timescale for condensation and evaporation.
                Defaults to None, in which case the timestep dt is used.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of constants. Defaults to None, in which case the
                parameters are obtained from the equation.
        """

        self.explicit_only = True
        label_name = 'saturation_adjustment'
        super().__init__(equation, label_name, parameters=parameters)

        self.time_varying_saturation = time_varying_saturation
        self.convective_feedback = convective_feedback
        self.thermal_feedback = thermal_feedback
        self.time_varying_gamma_v = time_varying_gamma_v

        # Check for the correct fields
        assert vapour_name in equation.field_names, f"Field {vapour_name} does not exist in the equation set"
        assert cloud_name in equation.field_names, f"Field {cloud_name} does not exist in the equation set"

        if self.convective_feedback:
            assert "D" in equation.field_names, "Depth field must exist for convective feedback"
            assert beta1 is not None, "If convective feedback is used, beta1 parameter must be specified"

        if self.thermal_feedback:
            assert "b" in equation.field_names, "Buoyancy field must exist for thermal feedback"
            assert beta2 is not None, "If thermal feedback is used, beta2 parameter must be specified"

        # Obtain function spaces
        W = equation.function_space
        self.Vv_idx = equation.field_names.index(vapour_name)
        self.Vc_idx = equation.field_names.index(cloud_name)
        Vv = W.sub(self.Vv_idx)
        Vc = W.sub(self.Vc_idx)
        # order for V_idxs is vapour, cloud
        V_idxs = [self.Vv_idx, self.Vc_idx]

        # depth needed if convective feedback
        if self.convective_feedback:
            self.VD_idx = equation.field_names.index("D")
            VD = W.sub(self.VD_idx)
            self.D = Function(VD)
            # order for V_idxs is now vapour, cloud, depth
            V_idxs.append(self.VD_idx)

        # buoyancy needed if thermal feedback
        if self.thermal_feedback:
            self.Vb_idx = equation.field_names.index("b")
            Vb = W.sub(self.Vb_idx)
            self.b = Function(Vb)
            # order for V_idxs is now vapour, cloud, depth, buoyancy
            V_idxs.append(self.Vb_idx)

        # tau is the timescale for condensation/evaporation (may or may not be the timestep)
        R = FunctionSpace(equation.domain.mesh, "R", 0)
        if tau is not None:
            self.set_tau_to_dt = False
            self.tau = Function(R).assign(tau)
        else:
            self.set_tau_to_dt = True
            self.tau = Function(R)
            logger.info("Timescale for moisture conversion between vapour and cloud has been set to dt. If this is not the intention then provide a tau parameter as an argument to SWSaturationAdjustment.")

        if self.time_varying_saturation:
            if isinstance(saturation_curve, FunctionType):
                self.saturation_computation = saturation_curve
                self.saturation_curve = Function(Vv)
            else:
                raise NotImplementedError(
                    "If time_varying_saturation is True then saturation must be a Python function of at least one prognostic field.")
        else:
            assert not isinstance(saturation_curve, FunctionType), "If time_varying_saturation is not True then saturation cannot be a Python function."
            self.saturation_curve = saturation_curve

        # Saturation adjustment expression, adjusted to stop negative values
        self.water_v = Function(Vv)
        self.cloud = Function(Vc)
        sat_adj_expr = (self.water_v - self.saturation_curve) / self.tau
        sat_adj_expr = conditional(sat_adj_expr < 0,
                                   max_value(sat_adj_expr,
                                             -self.cloud / self.tau),
                                   min_value(sat_adj_expr,
                                             self.water_v / self.tau))

        # If gamma_v depends on variables
        if self.time_varying_gamma_v:
            if isinstance(gamma_v, FunctionType):
                self.gamma_v_computation = gamma_v
                self.gamma_v = Function(Vv)
            else:
                raise NotImplementedError(
                    "If time_varying_thermal_feedback is True then gamma_v must be a Python function of at least one prognostic field.")
        else:
            assert not isinstance(gamma_v, FunctionType), "If time_varying_thermal_feedback is not True then gamma_v cannot be a Python function."
            self.gamma_v = gamma_v

        # Factors for multiplying source for different variables
        # the order matches the order in V_idx (vapour, cloud, depth, buoyancy)
        factors = [self.gamma_v, -self.gamma_v]
        if convective_feedback:
            factors.append(self.gamma_v*beta1)
        if thermal_feedback:
            factors.append(self.gamma_v*beta2)

        # Add terms to equations and make interpolators
        # sources have the same order as V_idxs and factors
        self.source = Function(W)
        self.source_expr = [split(self.source)[V_idx] for V_idx in V_idxs]
        self.source_int = [self.source.subfunctions[V_idx] for V_idx in V_idxs]
        self.source_interpolate = [interpolate(sat_adj_expr*factor, source)
                                   for source, factor in zip(self.source_int, factors)]

        # test functions have the same order as factors and sources (vapour,
        # cloud, depth, buoyancy) so that the correct test function multiplies
        # each source term
        tests = [equation.tests[idx] for idx in V_idxs]

        # Add source terms to residual
        for test, source_val in zip(tests, self.source_expr):
            equation.residual += source_label(
                self.label(subject(test * source_val * dx, self.source), self.evaluate)
            )

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source_label term generated by the physics.

        Computes the phyiscs contributions to water vapour and cloud water at
        each timestep.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        if self.convective_feedback:
            self.D.assign(x_in.subfunctions[self.VD_idx])
        if self.thermal_feedback:
            self.b.assign(x_in.subfunctions[self.Vb_idx])
        if self.time_varying_saturation:
            self.saturation_curve.interpolate(self.saturation_computation(x_in))
        if self.set_tau_to_dt:
            self.tau.assign(dt)
        self.water_v.assign(x_in.subfunctions[self.Vv_idx])
        self.cloud.assign(x_in.subfunctions[self.Vc_idx])
        if self.time_varying_gamma_v:
            self.gamma_v.interpolate(self.gamma_v_computation(x_in))

        for interpolator, src in zip(self.source_interpolate, self.source_int):
            src.assign(assemble(interpolator))
        # If a source output is provided, assign the source term to it
        if x_out is not None:
            x_out.assign(self.source)
