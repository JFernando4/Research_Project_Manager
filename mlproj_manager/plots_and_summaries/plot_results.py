import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
# from project files
from mlproj_manager.file_management import aggregate_large_results, aggregate_results, get_names_for_parameter_sweep
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


def aggregate_runs(results: np.ndarray, aggregation_type: str):
    """
    Aggregates the results of multiple runs
    param results: numpy array with the results of multiple runs
    param aggregation_type: indicates how to aggregate the results
    returns: numpy array containing the aggregated results
    """
    if aggregation_type == "avg":
        return np.average(results, axis=-1)
    elif aggregation_type == "sum":
        return np.sum(results, axis=-1)
    elif aggregation_type == "none":
        return results
    else:
        raise ValueError("{0} is not a valid aggregation_type.".format(aggregation_type))


def handle_plot_display(plot_arguments):
    """
    Handles how to display the plot and then closes the plot.
    param plot_arguments: script arguments parsed by argparse
    returns: None
    """
    if plot_arguments.save_plot is not None:
        plt.savefig(plot_arguments.save_plot + ".svg", dpi=300)
    else:
        plt.show()
    plt.close()


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


def handle_line_plot(plot_arguments, results_dir: str, results_name: str):
    parameter_combinations = [item for item in plot_arguments.parameter_combination.split(",")]
    colors = np.array([item for item in plot_arguments.colors.split(",")], dtype=str).reshape(1, -1)

    for j, param_comb in enumerate(parameter_combinations):
        temp_dir = os.path.join(results_dir, param_comb)
        results = load_binned_results(results_dir=temp_dir,
                                      results_name=results_name,
                                      bin_size=plot_arguments.bin_size,
                                      bin_axis=plot_arguments.bin_axis,
                                      denominator=plot_arguments.denominator)

        plot_lines(results, colors[j])

    plt.ylim((float(lim) for lim in plot_arguments.ylims.split(",")))

    handle_plot_display(plot_arguments)


def handle_multi_epoch_plot(plot_arguments, results_dir: str, results_name: str):
    pass


def aggregate_run_results_for_parameter_sweep(results: np.ndarray, aggregation_method: str):
    """
    Aggregates the results of each run of a parameter combination. Then, computes the average and standard error over
    all the runs.
    param results: 2-dimensional numpy array with results
    param aggregation_method: string indicating how to aggregate the results
                                - avg: take the average over each run
                                - sum: take the sum over each sum
                                - last: use the last column in the results
    return: average (float), standard error (float)
    """

    assert len(results.shape) == 2, "There are too many dimensions in the array"
    if aggregation_method == "avg":
        temp_results = np.average(results, axis=-1)
    elif aggregation_method == "ste":
        temp_results = np.sum(results, axis=-1)
    elif aggregation_method == "last":
        temp_results = results[:, -1]
    else:
        raise ValueError("Invalid aggregation method: {0}".format(aggregation_method))

    return compute_average_and_standard_error(temp_results)


def handle_parameter_sweep_plot(plot_arguments, results_dir: str, results_name: str, verbose=False):
    parameter_combinations = [item for item in plot_arguments.parameter_combination.split(",")]
    colors = np.array([item for item in plot_arguments.colors.split(",")], dtype=str)

    for i, param_comb in enumerate(parameter_combinations):

        assert isinstance(param_comb, str)
        temp_param_comb, temp_param_vals = get_names_for_parameter_sweep(param_comb,
                                                                         results_dir,
                                                                         return_parameter_values=True)
        x_axis = np.array(temp_param_vals, dtype=np.float32)
        avg_param_performance = np.zeros(len(temp_param_comb), dtype=np.float32)
        ste_param_performance = np.zeros(len(temp_param_comb), dtype=np.float32)

        for j, tpc in enumerate(temp_param_comb):
            results = load_binned_results(results_dir=os.path.join(results_dir, tpc),
                                          results_name=results_name,
                                          bin_size=plot_arguments.bin_size,
                                          bin_axis=plot_arguments.bin_axis,
                                          denominator=plot_arguments.denominator)

            avg, ste = aggregate_run_results_for_parameter_sweep(results, plot_arguments.run_aggregation_method)
            avg_param_performance[j] += avg
            ste_param_performance[j] += ste

            if verbose:
                print("Parameter Combination: {0}\n\tAverage: {1:.4f}\n\tStandard Error: {2:.4f}".format(tpc, avg, ste))

        line_plot_with_error_bars(results=avg_param_performance,
                                  error=ste_param_performance,
                                  x_axis=x_axis,
                                  color=colors[i],
                                  light_color=lighten_color(colors[i], 0.25))

    plt.ylim((float(lim) for lim in plot_arguments.ylims.split(",")))
    handle_plot_display(plot_arguments)


def parse_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--plot-type", action="store", type=str, default="line",
                           choices=["line", "multi_epoch", "parameter_sweep"])
    arguments.add_argument("--results-dir", action="store", type=str, required=True,
                           help="Path to the directory that contains the directories of each different parameter "
                                "combination.")
    arguments.add_argument("--parameter-combination", action="store", type=str, default="optimizer-sgd_stepsize-0.003",
                           help="Comma separated list. Each entry has the form parameter1-val1_parameter2_val2_....")
    arguments.add_argument("--results-name", action="store", type=str, required=True,
                           help="Name of the type of result. For example: ")
    arguments.add_argument("--run-aggregation-method", type=str, default="avg",
                           choices=["avg",          # average over whole run
                                    "sum",          # sum over whole run
                                    "none",         # do nothing
                                    "last"],        # last result in the run
                           help="Indicates how to aggregate the results for each run.")
    arguments.add_argument("--bin-size", action="store", type=int, default=100)
    arguments.add_argument("--bin-axis", action="store", type=int, default=1)
    arguments.add_argument("-c", "--colors", type=str, default="tab:blue", action="store", help="comma separated list")
    arguments.add_argument("-d", "--denominator", type=float, default=1, action="store",
                           help="Comma separated list. Results are divided by this number")
    arguments.add_argument("-el", "--epoch-length", type=int, default=None, required=False)
    arguments.add_argument("-ne", "--number-of-epochs", type=int, default=None, required=False)
    arguments.add_argument("-elist", "--epoch-list", action="store", type=str, default=None,
                           help="comma separated list", required=False)
    arguments.add_argument("-ylims", type=str, default="0.0,1.0", help="comma separated list")
    arguments.add_argument("-col", type=int, default=-1)
    arguments.add_argument("-sp", "--save-plot", action="store", type=str, default=None)
    arguments.add_argument("--verbose", action="store_true", default=False)
    return arguments.parse_args()


def main():
    plot_args = parse_arguments()
    results_dir = plot_args.results_dir
    results_name = plot_args.results_name
    plot_type = plot_args.plot_type

    if plot_type == "line":
        handle_line_plot(plot_args, results_dir, results_name)
    elif plot_type == "multi_epoch":
        handle_multi_epoch_plot(plot_args, results_dir, results_name)
    elif plot_type == "parameter_sweep":
        handle_parameter_sweep_plot(plot_args, results_dir, results_name, verbose=plot_args.verbose)
    else:
        raise ValueError

    # experiment_names = [item for item in plot_args.paramter_combination.split(",")]
    # bin_size = plot_args.bin_size
    # plot_type = plot_args.plot_type
    # colors = np.array([item for item in plot_args.colors.split(",")], dtype=str).reshape(1, -1)
    # denominators = [float(item) for item in plot_args.denominator.split(",")]
    # if len(denominators) == 1:
    #     denominators *= len(experiment_names)
    # if len(experiment_names) > 1: colors = colors.T
    #
    # for j, exp_name in enumerate(experiment_names):
    #
    #     temp_dir = os.path.join(results_dir, exp_name)
    #     results = load_binned_results(temp_dir, results_name, bin_size, plot_args.bin_axis,denominators[j])
    #     if plot_args.col > -1: results = results[:,:,plot_args.col]
    #     elif plot_args.col == -2: results = np.average(results, axis=-1)
    #     elif plot_args.col == -3: results = np.sum(results, axis=-1)
    #
    #     if plot_type == "line":
    #         plot_lines(results, colors[j])
    #     elif plot_type == "multi_epoch":
    #         if plot_args.epoch_list is not None:
    #             epoch_list = [int(e) for e in plot_args.epoch_list.split(",")]
    #             multi_epoch_plot(results=results, colors=colors[j], epoch_list=epoch_list,
    #                              epoch_length=plot_args.epoch_length)
    #         else:
    #             multi_epoch_plot(results=results, colors=colors[j], number_of_epochs=plot_args.number_of_epochs,
    #                              epoch_length=plot_args.epoch_length)
    #     else:
    #         raise ValueError("{0} is not a valid plot type!".format(plot_type))
    #
    # plt.ylim((float(lim) for lim in plot_args.ylims.split(",")))
    # if plot_args.save_plot is not None:
    #     plt.savefig(plot_args.save_plot + ".svg", dpi=300)
    # else:
    #     plt.show()
    # plt.close()


if __name__ == '__main__':
    main()
