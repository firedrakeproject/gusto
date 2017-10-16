from netCDF4 import Dataset


class Plotting(object):

    def __init__(self, filename, field_name, stats=None):

        # get data from file
        try:
            self.data = Dataset(filename, "r")
        except IOError:
            raise ValueError("File %s does not exist" % filename)

        # get time variable
        self.time = list(self.data.variables["time"])

        # get field
        try:
            self.grp = self.data.groups[field_name]
        except KeyError:
            raise ValueError("Field named %s does not exist in this file. You have these fields: %s" % (field_name, [str(f) for f in self.data.groups.keys()]))

        if stats is None:
            self.field = self.grp.variables[field_name]
        else:
            self.stats = []
            for stat in stats:
                try:
                    self.stats.append(self.grp.variables[stat])
                except KeyError:
                    raise ValueError("Stat named %s does not exist for this field. You have these stats: %s" % (stat, [str(s) for s in self.grp.variables.keys()]))
