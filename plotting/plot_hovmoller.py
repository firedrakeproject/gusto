import argparse
from plot_1D_profile import Plot1DProfile
import matplotlib.pyplot as plt


class Hovmoller(Plot1DProfile):

    def plot(self):

        plt.contour(self.px[:], self.time, self.f[:])
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="make Hovmoller plot of field with x axis specified by dim and other spatial dimensions set to values corresponding to indices in dim_idxs")
    parser.add_argument("filename", help="path to .nc file containing data")
    parser.add_argument("field_name", help="name of field to be plotted")
    parser.add_argument("dim", type=int, help="index of dimension to plot on x axis. 0 corresponds to the first dimension in your pointdata.nc file.")
    parser.add_argument("val", type=float, nargs="+", help="value of other dimension")

    args = parser.parse_args()

    hov = Hovmoller(args.filename, args.field_name,
                    args.dim, args.val)
    hov.plot()
