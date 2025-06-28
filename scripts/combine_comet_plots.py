import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np
import re

plt.rcParams['figure.dpi'] = 300
sns.set_style("darkgrid")

"""
Download the JSON files from Comet.ml containing the plots' data
and run this script with the json files as arguments to combine
them into a single plot.

Optional things you can do to customize the plot:
- "--title" to set the title of the plot.
- "--max_x" to set the maximum x-value to plot.
- "--min_x" to set the minimum x-value to plot.
- "--max_y" to set the maximum y-value to plot.
- "--min_y" to set the minimum y-value to plot.
- "--xlabel" to set the label for the x-axis.
- "--ylabel" to set the label for the y-axis.
- "--output-file" to save the plot to a file instead of displaying it.

Additionnal things you can do INSIDE THE JSON FILES:
- Add an "offset" key to an item to add an offset to its x-values.
- Add a "min_x" key to an item to remove all x-values less than min_x.
- Add a "max_x" key to an item to remove all x-values greater than max_x.
- Add a "legend" key to an item to set the label in the legend.
"""

def read_json_files(json_paths):
    """Reads and parses multiple JSON files."""
    data = []
    for path in json_paths:
        with open(path, 'r') as file:
            data.extend(json.load(file))
    return data

def plot_data(data, args, output_file=None):
    """Plots the data as is. Saves or shows the plot."""

    # Extract arguments
    focus_x = (args.min_x, args.max_x) if args.min_x or args.max_x else None
    focus_y = (args.min_y, args.max_y) if args.min_y or args.max_y else None
    title = args.title
    xlabel = args.xlabel
    ylabel = args.ylabel


    plt.figure(figsize=(10, 6))

    # If average_reg is specified, we want to average the data that matches the regular expression.
    # The regular expression is "average_reg". We want to check all items "name" that matches
    # that expression and average their data and plot it as a line.
    if args.average_reg:
        x = data[0]['x']
        print("X_length: ", len(x))
        average_data_y = []
        for item in data:
            if re.match(args.average_reg, item['name']):
                average_data_y.append(item['y'])
                print("Matched {} with length {}".format(item['name'], len(item['x'])))

                assert len(item['x']) == len(x), "All x-values must have the same length."
        
        average_data_y = np.mean(average_data_y, axis=0)
        std_data_y = np.std(average_data_y, axis=0)
        plt.plot(x, average_data_y, label='Average', alpha=0.6)
        plt.fill_between(x, average_data_y - std_data_y, average_data_y + std_data_y, alpha=0.2, color='blue')

    # Plot other data
    for item in data:
        label = item.get("legend", item.get("name", None))
        plt.plot(item['x'], item['y'], label=label, alpha=0.6)

        if item.get("dotted", None):
            plt.axhline(item.get("dotted"), color='red', linestyle='--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    # If focus_x is specified, we want to "zoom" into the plot so that
    # we only see values from focus_x[0] to focus_x[1] on the x-axis.
    if focus_x:
        print("setting limit to x-axis: ", focus_x)
        plt.xlim(left=focus_x[0], right=focus_x[1])

    # If focus_y is specified, we want to "zoom" into the plot so that
    # we only see values from focus_y[0] to focus_y[1] on the y-axis.
    if focus_y:
        print("setting limit to y-axis: ", focus_y)
        plt.ylim(bottom=focus_y[0], top=focus_y[1])

    if output_file:
        plt.savefig(output_file)
        print(f"Figure saved to {output_file}")
    else:
        plt.show()

def process_data(data):
    # If "offset" key is present, add the offset to the x-value of that item.
    for item in data:
        if "offset" in item:
            item["x"] = [x + item["offset"] for x in item["x"]]
    
    # If "min_x" is present, remove all x-values less than min_x.
    # If "max_x" is present, remove all x-values greater than max_x.
    bounds_keys = ["min_x", "max_x"]
    for item in data:
        bounds = [item[key] for key in bounds_keys if key in item]
        if bounds:
            min_x, max_x = bounds
            item["x"], item["y"] = zip(*[(x, y) for x, y in zip(item["x"], item["y"]) if min_x <= x <= max_x])

    return data

def main():
    parser = argparse.ArgumentParser(description="Combine and plot multiple JSON files.")
    parser.add_argument(
        "json_files", 
        nargs="+", 
        help="Paths to the JSON files to combine and plot."
    )
    parser.add_argument("--title", type=str, default="Combined Line Plot", help="Title of the plot.")
    parser.add_argument("--max_x", type=float, default=None, help="Maximum x-value to plot.")
    parser.add_argument("--min_x", type=float, default=None, help="Minimum x-value to plot.")
    parser.add_argument("--max_y", type=float, default=None, help="Maximum y-value to plot.")
    parser.add_argument("--min_y", type=float, default=None, help="Minimum y-value to plot.")
    parser.add_argument("--xlabel", type=str, default="X-axis", help="Label for the x-axis.")
    parser.add_argument("--ylabel", type=str, default="Y-axis", help="Label for the y-axis.")
    parser.add_argument("--average_reg", type=str, default=None, help="Regular expression to average the data.")
    
    parser.add_argument(
        "--output-file", 
        type=str, 
        default=None, 
        help="Path to save the plot. If not specified, the plot is displayed."
    )
    args = parser.parse_args()

    # Read and plot data
    data = read_json_files(args.json_files)
    data = process_data(data)
    plot_data(data, args, output_file=args.output_file)

if __name__ == "__main__":
    main()