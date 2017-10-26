import argparse
from plotting import Plotting
import matplotlib.pyplot as plt


class PlotStats(Plotting):

    def __init__(self, filename, field_name, stats, normalise=False):
        super().__init__(filename, field_name, stats)
        self.normalise = normalise

    def plot(self, same_plot):

        for stat in self.stats:
            if self.normalise:
                s = (stat - stat[0])/stat[0]
            else:
                s = stat
            plt.plot(self.time, s, label=stat.name)
            if not same_plot:
                plt.xlabel("time")
                plt.show()

        if same_plot:
            plt.xlabel("time")
            plt.legend(loc='upper left')
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="plot stats over time for specified field")
    parser.add_argument("filename", help="path to .nc file containing data")
    parser.add_argument("field_name", help="name of field")
    parser.add_argument("stats", nargs="+", help="names of stats to plot")
    parser.add_argument("--same_plot", action="store_true")
    parser.add_argument("--normalise", action="store_true")
    args = parser.parse_args()
    pltstats = PlotStats(args.filename, args.field_name,
                         args.stats, args.normalise)
    pltstats.plot(args.same_plot)
