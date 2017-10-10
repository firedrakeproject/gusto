import argparse
from plotting import Plotting
import matplotlib.pyplot as plt
import numpy as np


class Plot1DProfile(Plotting):

    def __init__(self, filename, field_name, dim, val, time_entries=None):

        super().__init__(filename, field_name)

        # if no time_entries are specified, plot them all
        if time_entries is not None:
            # check that no time entries exceed the length of the time dimension
            if any(i > len(self.time) for i in time_entries):
                raise ValueError("You cannot plot a time entry greater than %i" % len(self.time))

            self.times = []
            for i in time_entries:
                self.times.append(self.time[i])
            print("Profile will be plotted for times %s :" % [str(t) for t in self.times])
        else:
            time_entries = [i for i in range(len(self.time))]
            self.times = self.time

        # get points
        points = self.grp.variables["points"]

        # check that user has fixed the values of all other dimensions
        if len(val) != points.shape[1]-1:
            raise ValueError("You must fix the values of exactly %s dimensions." % str(points.shape[1]-1))

        # find out which points satisfy requirements
        self.px = []
        idx = []
        for i, p in enumerate(points):
            pq = np.delete(p, [dim])
            if pq == val:
                self.px.append(p[dim])
                idx.append(i)

        if len(idx) == 0:
            raise RuntimeError("No points match your requirements")

        # field values
        self.f = self.field[time_entries, idx]

    def plot(self, same_plot):

        for i in range(len(self.times)):
            plt.plot(self.px, self.f[i])
            if not same_plot:
                plt.show()

        if same_plot:
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="plot 1D profile of field with x axis specified by dim and other spatial dimensions set to values corresponding to indices in dim_idxs")
    parser.add_argument("filename", help="path to .nc file containing data")
    parser.add_argument("field_name", help="name of field to be plotted")
    parser.add_argument("dim", type=int, help="index of dimension to plot on x axis. 0 corresponds to the first dimension in your pointdata.nc file.")
    parser.add_argument("val", type=float, nargs="+", help="value of other dimension")
    parser.add_argument("--time_entries", type=int, nargs="+",
                        help="integers specifying the time entries at which to plot data")
    parser.add_argument("--same_plot", action="store_true")
    args = parser.parse_args()

    plt1D = Plot1DProfile(args.filename, args.field_name,
                          args.dim, args.val, args.time_entries)
    plt1D.plot(args.same_plot)
