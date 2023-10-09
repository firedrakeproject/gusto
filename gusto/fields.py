from firedrake import Function, MixedElement, functionspaceimpl

__all__ = ["PrescribedFields", "TimeLevelFields", "StateFields"]


class Fields(object):
    """Object to hold and create a specified set of fields."""
    def __init__(self, equation):
        """
        Args:
            equation (:class:`PrognosticEquation`): an equation object.
        """
        self.fields = []
        subfield_names = equation.field_names if hasattr(equation, "field_names") else None
        self.add_field(equation.field_name, equation.function_space,
                       subfield_names)

    def add_field(self, name, space, subfield_names=None):
        """
        Adds a new field to the :class:`Fields` object.
        Args:
            name (str): the name of the prognostic variable.
            space (:class:`FunctionSpace`): the space to create the field in.
                Can also be a :class:`MixedFunctionSpace`, and then will create
                the fields in their appropriate subspaces. Then, the
                'subfield_names' argument must be provided.
            subfield_names (list, optional): a list of names of the prognostic
                variables. Defaults to None.
        """

        value = Function(space, name=name)
        setattr(self, name, value)
        self.fields.append(value)

        if len(space) > 1:
            assert len(space) == len(subfield_names)
            for field_name, field in zip(subfield_names, value.subfunctions):
                setattr(self, field_name, field)
                field.rename(field_name)
                self.fields.append(field)

    def __call__(self, name):
        """
        Returns a specified field from the :class:`Fields` object.
        Args:
            name (str): the name of the field.
        Returns:
            :class:`Function`: the desired field.
        """
        return getattr(self, name)

    def __iter__(self):
        """Returns an iterable of the contained fields."""
        return iter(self.fields)


class PrescribedFields(Fields):
    """Object to hold and create a specified set of prescribed fields."""
    def __init__(self):
        self.fields = []
        self._field_names = []

    def __call__(self, name, space=None):
        """
        Returns a specified field from the :class:`PrescribedFields`. If a named
        field does not yet exist in the :class:`PrescribedFields` object, then
        the space argument must be specified so that it can be created and added
        to the object.

        Args:
            name (str): the name of the field.
            space (:class:`FunctionSpace`, optional): the function space to
                create the field in. Defaults to None.

        Returns:
            :class:`Function`: the desired field.
        """
        if hasattr(self, name):
            # Field already exists in object, so return it
            return getattr(self, name)
        else:
            # Create field
            self.add_field(name, space)
            self._field_names.append(name)
            return getattr(self, name)

    def __iter__(self):
        """Returns an iterable of the contained fields."""
        return iter(self.fields)


class StateFields(Fields):
    """
    Container for all of the model's fields.

    The `StateFields` are a container for all the fields to be used by a time
    stepper. In the case of the prognostic fields, these are pointers to the
    time steppers' fields at the (n+1) time level. Prescribed fields are
    pointers to the respective equation sets, while diagnostic fields are
    created here.
    """

    def __init__(self, prognostic_fields, prescribed_fields, *fields_to_dump):
        """
        Args:
            prognostic_fields (:class:`Fields`): the (n+1) time level fields.
            prescribed_fields (iter): an iterable of (name, function_space)
                tuples, that are used to create the prescribed fields.
            *fields_to_dump (str): the names of fields to be dumped.
        """
        self.fields = []
        output_specified = len(fields_to_dump) > 0
        self.to_dump = sorted(set((fields_to_dump)))
        self.to_pick_up = []
        self._field_types = []
        self._field_names = []

        # Add pointers to prognostic fields
        for field in prognostic_fields.np1.fields:
            # Don't add the mixed field
            if type(field.ufl_element()) is not MixedElement:
                # If fields_to_dump not specified, dump by default
                to_dump = field.name() in fields_to_dump or not output_specified
                self.__call__(field.name(), field=field, dump=to_dump,
                              pick_up=True, field_type="prognostic")
            else:
                self.__call__(field.name(), field=field, dump=False,
                              pick_up=False, field_type="prognostic")

        # For multi-level schemes, previous time levels need adding to the state
        if len(prognostic_fields.levels) > 2:
            for level in range(len(prognostic_fields.levels)-2):
                level_name = 'n' if level == 0 else f'nm{level}'
                field_suffix = f'nm{level+1}'
                # Add pointers to prognostic fields, don't add the mixed field
                for field in getattr(prognostic_fields, level_name).fields:
                    if type(field.ufl_element()) is not MixedElement:
                        self.__call__(f'{field.name()}_{field_suffix}', field=field,
                                      dump=False, pick_up=True, field_type="prognostic")

        # Add pointers to any prescribed fields
        for field in prescribed_fields.fields:
            to_dump = field.name() in fields_to_dump
            self.__call__(field.name(), field=field, dump=to_dump,
                          pick_up=True, field_type="prescribed")

    def __call__(self, name, field=None, space=None, dump=True, pick_up=False,
                 field_type=None):
        """
        Returns a field from or adds a field to the :class:`StateFields`.

        If a named field does not yet exist in the :class:`StateFields`, then
        the optional arguments must be specified so that it can be created and
        added to the :class:`StateFields`.

        If "field" is specified, then the pointer to the field is added to the
        :class:`StateFields` object. If "space" is specified, then the field
        itself is created.

        Args:
            name (str): name of the field to be returned/added.
            field (:class:`Function`, optional): an existing field to be added
                to the :class:`StateFields` object.
            space (:class:`FunctionSpace`, optional): the function space to
                create the field in. Defaults to None.
            dump (bool, optional): whether the created field should be
                outputted. Defaults to True.
            pick_up (bool, optional): whether the created field should be picked
                up when checkpointing. Defaults to False.

        Returns:
            :class:`Function`: the specified field.
        """
        if hasattr(self, name):
            # Field already exists in object, so return it
            return getattr(self, name)
        else:
            # Field does not yet exist in StateFields
            if field is None and space is None:
                raise ValueError(f'Field {name} does not exist in StateFields. '
                                 + 'Either field or space argument must be '
                                 + 'specified to add this field to StateFields')
            elif field is not None and space is not None:
                raise ValueError('Cannot specify both field and space to StateFields')

            if field is not None:
                # Field pointer, so just add existing field to StateFields
                assert isinstance(field, Function), \
                    f'field argument for creating field {name} must be a Function, not {type(field)}'
                setattr(self, name, field)
                self.fields.append(field)
            else:
                # Create field
                assert isinstance(space, functionspaceimpl.WithGeometry), \
                    f'space argument for creating field {name} must be FunctionSpace, not {type(space)}'
                self.add_field(name, space)

            if dump:
                self.to_dump.append(name)
            if pick_up:
                self.to_pick_up.append(name)

            # Work out field type
            if field_type is None:
                # Prognostics can only be specified through __init__
                if pick_up:
                    field_type = "prescribed"
                elif dump:
                    field_type = "diagnostic"
                else:
                    field_type = "derived"
            else:
                permitted_types = ["prognostic", "prescribed", "diagnostic",
                                   "derived", "reference"]
                assert field_type in permitted_types, \
                    f'field_type {field_type} not in permitted types {permitted_types}'
            self._field_types.append(field_type)
            self._field_names.append(name)

            return getattr(self, name)

    def field_type(self, field_name):
        """
        Returns the type (e.g. prognostic/diagnostic) of a field held in the
        :class:`StateFields`.

        Args:
            field_name (str): name of the field to return the type of.

        Returns:
            str: a string describing the type (e.g. prognostic) of the field.
        """
        assert hasattr(self, field_name), f'StateFields has no field {field_name}'
        idx = self._field_names.index(field_name)
        return self._field_types[idx]


class TimeLevelFields(object):
    """Creates the fields required in the :class:`Timestepper` object."""

    def __init__(self, equation, nlevels=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): an equation object.
            nlevels (optional, iterable): an iterable containing the names
                of the time levels
        """
        default_levels = ("n", "np1")
        if nlevels is None or nlevels == 1:
            previous_levels = []
        else:
            previous_levels = ["nm%i" % n for n in range(nlevels-1, 0, -1)]
        levels = tuple(previous_levels) + default_levels
        self.levels = levels

        self.add_fields(equation, levels)
        self.previous = [getattr(self, level) for level in previous_levels]
        self.previous.append(getattr(self, "n"))

    def add_fields(self, equation, levels=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): an equation object.
            levels (iterable, optional): an iterable containing the names
                of the time levels to be added. Defaults to None.
        """
        if levels is None:
            levels = self.levels
        for level in levels:
            try:
                x = getattr(self, level)
                x.add_field(equation.field_name, equation.function_space)
            except AttributeError:
                setattr(self, level, Fields(equation))

    def initialise(self, state_fields):
        """
        Initialises the time fields from those currently in the equation.

        Args:
            state_fields (:class:`StateFields`): the model's field container.
        """
        for field in self.n:
            field.assign(state_fields(field.name()))
            self.np1(field.name()).assign(field)

    def update(self):
        """Updates the fields, copying the values to previous time levels"""
        for i in range(len(self.previous)-1):
            xi = self.previous[i]
            xip1 = self.previous[i+1]
            for field in xi:
                field.assign(xip1(field.name()))
        for field in self.n:
            field.assign(self.np1(field.name()))
