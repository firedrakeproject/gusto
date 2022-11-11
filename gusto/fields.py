from firedrake import Function

__all__ = ["TimeLevelFields", "StateFields"]


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
        Adds a new field to the :class:`FieldCreator`.

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
            for field_name, field in zip(subfield_names, value.split()):
                setattr(self, field_name, field)
                field.rename(field_name)
                self.fields.append(field)

    def __call__(self, name):
        """
        Returns a specified field from the :class:`FieldCreator`.

        Args:
            name (str): the name of the field.

        Returns:
            :class:`Function`: the desired field.
        """
        return getattr(self, name)

    def __iter__(self):
        """Returns an iterable of the contained fields."""
        return iter(self.fields)


class StateFields(Fields):
    """Creates the prognostic fields for the :class:`State` object."""

    def __init__(self, *fields_to_dump):
        """
        Args:
            *fields_to_dump (str): the names of fields to be dumped.
        """
        self.fields = []
        self.output_specified = len(fields_to_dump) > 0
        self.to_dump = set((fields_to_dump))
        self.to_pickup = set(())

    def __call__(self, name, space=None, subfield_names=None, dump=True,
                 pickup=False):
        """
        Returns a field from or adds a field to the :class:`StateFields`.

        If a named field does not yet exist in the :class:`StateFields`, then
        the optional arguments must be specified so that it can be created and
        added to the :class:`StateFields`.

        Args:
            name (str): name of the field to be returned/added.
            space (:class:`FunctionSpace`, optional): the function space to
                create the field in. Defaults to None.
            subfield_names (list, optional): a list of names of the constituent
                prognostic variables to be created, if the provided space is
                actually a :class:`MixedFunctionSpace`. Defaults to None.
            dump (bool, optional): whether the created field should be
                outputted. Defaults to True.
            pickup (bool, optional): whether the created field should be picked
                up when checkpointing. Defaults to False.

        Returns:
            :class:`Function`: the specified field.
        """
        try:
            return getattr(self, name)
        except AttributeError:
            self.add_field(name, space, subfield_names)
            if dump:
                if subfield_names is not None:
                    self.to_dump.update(subfield_names)
                else:
                    self.to_dump.add(name)
            if pickup:
                if subfield_names is not None:
                    self.to_pickup.update(subfield_names)
                else:
                    self.to_pickup.add(name)
            return getattr(self, name)


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
            previous_levels = ["nm1"]
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

    def initialise(self, state):
        """
        Initialises the time fields from those currently in state

        Args: state (:class:`State`): the model state object
        """
        for field in self.n:
            field.assign(state.fields(field.name()))
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
