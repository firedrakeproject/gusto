import argparse
from plotting import Plotting
import matplotlib.pyplot as plt

class Plot1DProfile(Plotting):

    def __init__(self, filename, field_name, dim, dim_idxs, time_entries=None):

        super().__init__(filename, field_name)

        # if no time_entries are specified, plot them all
        if time_entries is not None:
            # check that no time entries exceed the length of the time dimension
            if any(i > len(self.time) for i in time_entries):
                raise ValueError("You cannot plot a time entry greater than %i" % len(time))

            self.times = []
            for i in time_entries:
                self.times.append(self.time[i])
            print("Profile will be plotted for times %s :" % [str(t) for t in self.times ])
        else:
            self.times = self.time

        # check that dim is a valid dimension for this data
        ndims = len(self.grp.dimensions.keys())
        if ndims < 2 or ndims > 3:
            raise RuntimeError("I do not know what to do with data of dimension %i" % ndims)
        if dim > ndims-1:
            raise ValueError("Specified dim %i is greater than the number of spatial dimensions of this data %i (remember that dim=0 corresponds to the first dimension)." % (dim, ndims))
        if len(dim_idxs) != ndims-1:
            raise ValueError("You must specify the indices of %i dimensions." % ndims-1)

        # set up slicing tuple (currently a list so we can append to it)
        # first dimension is time, which we have already dealt with so
        # we want all of these values
        obj = [slice(None, None, 1)]

        # loop over dimensions and either append index for this
        # dimension as specified by the user in dim_idxs, or, if we
        # have reached the dimension to plot over, as specified by the
        # dim option, then take all of the values
        j = 0
        for i in range(ndims):
            if i != dim:
                if dim_idxs[j] > len(self.grp.variables["x"+str(i)])-1:
                    raise ValueError("Chosen index exceeds length of corresponding dimension")
                obj.append(dim_idxs[j])
                j += 1
            else:
                obj.append(slice(None, None, 1))
                idx = "x"+str(dim)
        # field values
        self.f = self.field[tuple(obj)]
        # points
        self.px = self.grp.variables[idx]

    def plot(self, same_plot):

        for i in range(len(self.times)):
            plt.plot(self.px, self.f[i])
            if not same_plot:
                plt.show()

        if same_plot:
                plt.show()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="plot 1D profile of field with x axis specified by dim and other spatial dimensions set to values corresponding to indices in dim_idxs")
    parser.add_argument("filename", help="path to .nc file containing data")
    parser.add_argument("field_name", help="name of field to be plotted")
    parser.add_argument("time_entries", type=int, nargs="+",
                        help="integers specifying the time entries at which to plot data")
    parser.add_argument("dim", type=int, help="index of dimension to plot on x axis. 0 corresponds to the first dimension in your pointdata.nc file.")
    parser.add_argument("--dim_idxs", type=int, nargs="+")
    parser.add_argument("--same_plot", action="store_true")
    args = parser.parse_args()
    print(args)
    plt1D = Plot1DProfile(args.filename, args.field_name,
                            args.time_entries, args.dim, args.dim_idxs)
    plt1D.plot(args.same_plot)
