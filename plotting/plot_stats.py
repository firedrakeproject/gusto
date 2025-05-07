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
            plt.plot(self.time, s[:], label=stat.name)
            if not same_plot:
                plt.xlabel("time")
                plt.show()

        if same_plot:
            plt.xlabel("time (days)", fontsize="12")
            # xticks = [0, 864000, 1728000, 2592000, 3456000, 4320000]
            # xlabels = [0, 10, 20, 30, 40, 50]
            xticks = [0, 432000, 864000, 1296000]
            xlabels = [0, 5, 10, 15]
            plt.xticks(xticks, labels=xlabels, fontsize="12")
            plt.yticks(fontsize="12")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.ylabel("energy conservation error", fontsize="12")
            # plt.legend(loc='upper left')
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
