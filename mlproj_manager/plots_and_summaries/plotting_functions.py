import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mc
import colorsys
# from project files:
from mlproj_manager.plots_and_summaries.summaries import compute_average_and_standard_error


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    :param color: (matplotlib color string, hex string, or RGB tuple) color to make lighter or darker
    :param amount: (int) number between 0.0 and 1.7 to determine how much lighter or darker to make the color
                         0.0 is lighter, 1.7 is darker, 1.0 no change
    """
    if color in mc.cnames.keys():
        c = mc.cnames[color]
    else:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def color_fader(c1,c2,mix=0.0):
    """
    fades (linear interpolates) from color c1 (at mix=0) to c2 (mix=1)
    :param c1: (string, hex code) initial color
    :param c2: (string, hex code) final color
    :param mix: (float between 0 and 1) degree of interpolation
    :return: a string with the hex code of the new color
    """
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def parse_plot_kwarg(keyword_dict: dict):
    """
    Parses the keyword arguments for plotting functions
    :param keyword_dict: (dict) dictionary with the keyword arguments
    :return: (dict) with all  the necessary keywords plotting, any missing keyword in keyword_dict is set to a default
                    value
    """
    keys = keyword_dict.keys()
    parameter_list = [("color", "#000000"), ("linewidth", 1), ("light_color", "#DFDFDF"), ("linestyle", "solid"),
                      ("label", None), ("alpha", 1.0)]
    arguments = {
        parameter_name: default_value if parameter_name not in keys else keyword_dict[parameter_name]
        for parameter_name, default_value in parameter_list
    }
    return arguments


def line_plot_with_error_bars(results: np.ndarray, error: np.ndarray, x_axis=None, **kwargs):
    """
    Creates a line plot with a shaded region of results ± error
    :param results: (np.array or list) array containing the results
    :param error: (np.array or list) array containing some error measure, eg., standard error
    :param x_axis: x_axis of the plot, should be same length as results or None
    :param kwargs: keyword arguments to be parsed by function parse_plot_kwarg
    :return: None, but it adds to the current matplotlib figure
    """
    assert results.size == error.size

    arguments = parse_plot_kwarg(kwargs)

    if x_axis is None:
        x_axis = np.arange(results.size)
    assert isinstance(x_axis, np.ndarray)
    assert x_axis.size == results.size

    plt.plot(x_axis, results, label=arguments["label"], color=arguments["color"], linewidth=arguments["linewidth"],
             linestyle=arguments["linestyle"], alpha=arguments["alpha"])
    plt.fill_between(x_axis, results - error, results + error, color=arguments["light_color"], alpha=arguments["alpha"])


def plot_lines(results: np.ndarray, colors: list, axis=-1):
    """
    Plots a line for each entry in array along a specified axis
    :param results: (np.ndarray) a 2-dimensional array with results
    :param colors: (list) a color for each different line
    :param axis: axis along which to plot the results
    :return: Noe, but it adds to the current matplotlib figure
    """
    results = results if len(results.shape) > 2 else results.reshape(results.shape + (1,))
    num_lines = results.shape[axis]
    for i in range(num_lines):
        avg, stderr = compute_average_and_standard_error(results[:, :, i], axis=0)
        line_plot_with_error_bars(avg, stderr, color=colors[i], light_color=lighten_color(colors[i], 0.25))
