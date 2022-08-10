import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
# from project files
from mlproj_manager.file_management import aggregate_large_results, aggregate_results
from mlproj_manager.plots_and_summaries.summaries import compute_average_and_standard_error
from mlproj_manager.plots_and_summaries.plotting_functions import line_plot_with_error_bars, lighten_color, color_fader


def load_binned_results(results_dir, results_name, bin_size, bin_axis, denominator):
    binned_results_name = results_name + "_bin-" + str(bin_size)
    results_path = os.path.join(results_dir, binned_results_name + ".npy")
    if not os.path.isfile(results_path) and bin_size > 0:
        results = aggregate_large_results(results_dir, results_name, bin_size, bin_axis=bin_axis)
    elif bin_size > 0:
        results = np.load(results_path)
    else:
        results = aggregate_results(results_dir, results_name, bin_size, bin_axis=bin_axis)
    results = np.float32(results) / denominator
    return results


def plot_lines(results: np.ndarray, colors: list):
    results = results if len(results.shape) > 2 else results.reshape(results.shape + (1,))
    num_lines = results.shape[-1]
    for i in range(num_lines):
        avg, stderr = compute_average_and_standard_error(results[:, :, i], axis=0)
        line_plot_with_error_bars(avg, stderr, color=colors[i], light_color=lighten_color(colors[i], 0.25))


def multi_epoch_plot(epoch_length: int, results: np.ndarray, colors: list, epoch_list=None,  number_of_epochs=None):
    # check that the dimensions are correct
    assert len(results.shape) == 2
    assert results.shape[1] % epoch_length == 0

    avg, stderr = compute_average_and_standard_error(results, axis=0)
    if epoch_list is None:
        assert number_of_epochs is not None
        new_shape = (avg.shape[0] // epoch_length, epoch_length)
        step = (avg.shape[0] // epoch_length) // number_of_epochs
        reshaped_avg = avg.reshape(new_shape)[::step, :]
        reshaped_stderr = stderr.reshape(new_shape)[::step, :]
        alpha = 0.98
        for i in range(number_of_epochs):
            temp_color = color_fader(colors[0], colors[1], mix=i / (number_of_epochs-1))
            line_plot_with_error_bars(reshaped_avg[i], reshaped_stderr[i], color=temp_color,
                                      light_color=lighten_color(temp_color, 0.25), alpha=alpha**i)
    else:
        for j, epoch in enumerate(epoch_list):
            start = epoch * epoch_length
            end = (epoch + 1) * epoch_length
            line_plot_with_error_bars(avg[start:end], stderr[start:end], color=colors[j],
                                      light_color=lighten_color(colors[j], 0.25))


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("-pt", "--plot_type", action="store", type=str, default="line")
    arguments.add_argument("-rd", "--results_dir", action="store", type=str, required=True)
    arguments.add_argument("-en", "--experiment_name", action="store", type=str, default="optimizer-sgd_stepsize-0.003",
                           help="comma separated list")
    arguments.add_argument("-rn","--results_name", action="store", type=str, required=True)
    arguments.add_argument("-bs", "--bin_size", action="store", type=int, default=100)
    arguments.add_argument("-ba", "--bin_axis", action="store", type=int, default=1)
    arguments.add_argument("-c", "--colors", type=str, default="tab:blue", action="store", help="comma separated list")
    arguments.add_argument("-d", "--denominator", type=str, default=1.0, action="store",
                           help="comma separated list, results are divided by this number")
    arguments.add_argument("-el", "--epoch_length", type=int, default=None, required=False)
    arguments.add_argument("-ne", "--number_of_epochs", type=int, default=None, required=False)
    arguments.add_argument("-elist", "--epoch-list", action="store", type=str, default=None,
                           help="comma separated list", required=False)
    arguments.add_argument("-ylims", type=str, default="0.0,1.0", help="comma separated list")
    arguments.add_argument("-col", type=int, default=-1)
    arguments.add_argument("-sp", "--save_plot", action="store", type=str, default=None)
    parsed_args = arguments.parse_args()

    results_dir = parsed_args.results_dir
    experiment_names = [item for item in parsed_args.experiment_name.split(",")]
    results_name = parsed_args.results_name
    bin_size = parsed_args.bin_size
    plot_type = parsed_args.plot_type
    colors = np.array([item for item in parsed_args.colors.split(",")], dtype=str).reshape(1, -1)
    denominators = [float(item) for item in parsed_args.denominator.split(",")]
    if len(denominators) == 1:
        denominators *= len(experiment_names)
    if len(experiment_names) > 1: colors = colors.T

    for j, exp_name in enumerate(experiment_names):

        temp_dir = os.path.join(results_dir, exp_name)
        results = load_binned_results(temp_dir, results_name, bin_size, parsed_args.bin_axis,denominators[j])
        if parsed_args.col > -1: results = results[:,:,parsed_args.col]
        elif parsed_args.col == -2: results = np.average(results, axis=-1)
        elif parsed_args.col == -3: results = np.sum(results, axis=-1)

        if plot_type == "line":
            plot_lines(results, colors[j])
        elif plot_type == "multi_epoch":
            if parsed_args.epoch_list is not None:
                epoch_list = [int(e) for e in parsed_args.epoch_list.split(",")]
                multi_epoch_plot(results=results, colors=colors[j], epoch_list=epoch_list,
                                 epoch_length=parsed_args.epoch_length)
            else:
                multi_epoch_plot(results=results, colors=colors[j], number_of_epochs=parsed_args.number_of_epochs,
                                 epoch_length=parsed_args.epoch_length)
        else:
            raise ValueError("{0} is not a valid plot type!".format(plot_type))

    plt.ylim((float(lim) for lim in parsed_args.ylims.split(",")))
    if parsed_args.save_plot is not None:
        plt.savefig(parsed_args.save_plot + ".svg", dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    main()
