"""Visualization utilities for metrics."""

from collections import OrderedDict
from typing import Dict, List, Optional

from matplotlib import pyplot as plt


def plot_boxes(indicators_list: List[Dict[str, List[float]]],
               indicator_names: Optional[List[str]] = None,
               indicator_colors: Optional[List[str]] = None) -> None:
    """Plot boxes from multiple indicators dicts.

    Parameters
    ----------
    indicators_list : List[Dict[str, List[float]]]
        List of indicators dicts. Each values should be 1D arrays
        representing an indictor value for each image. The keys are
        not necessarily the same over the different dicts.
    indicator_names : list of strs or None, optional
        List of names for the indicators. If None, the names will
        be the number of the indicator in the list.
    indicator_colors : list of strs or None, optional
        List of colors for the boxes representing the indicators.
        If None, the colors will
        be white for all indicators.

    Note
    ----
        - You can use ``plt.figure`` before calling this function to create \
        a new figure with properties you want.

        - You should run ``plt.show()`` to display the plot after calling \
        this function.

    Example
    -------
    .. code-block:: python3

        plt.figure(figsize=(15, 20))
        plot_boxes([indicators_1, indicators_2],
                    indicator_names=['facies1', 'facies2'],
                    indicator_colors=['lightblue', 'lightgreen'])
        plt.show()

    This plots the boxes for the two indicators dicts. Each different
    key in the indicators will correspond to a different subplot.
    The boxes for indicator_1 will be named 'facies1' and colored
    light blue, and the boxes for indicator_2 will be named 'facies2'
    and colored light green.
    """
    # Get all indicators names (preserving order)
    ind_names = [
        ind_name for indicators in indicators_list
        for ind_name in indicators.keys()
    ]
    ind_names = list(OrderedDict.fromkeys(ind_names))

    for i, ind_name in enumerate(ind_names):
        # values: list of values for the indicator ind_name
        #     for each indicators if exists
        # labels: names of the indicators that contains
        #     the indicator ind_name
        # colors: colors of the indicators that contains
        #    the indicator ind_name
        values, labels, colors = [], [], []
        for ind_dict, indicators in enumerate(indicators_list):
            if ind_name in indicators:
                # The indicator is in indicators : add the values
                values.append(indicators[ind_name])
                if indicator_names is not None:
                    labels.append(indicator_names[ind_dict])
                else:
                    labels.append(str(ind_dict + 1))  # + 1 to start at 1
                if indicator_colors is not None:
                    colors.append(indicator_colors[ind_dict])

        axi = plt.subplot(len(ind_names) // 4 + 1, 4, i + 1)
        # patch_artist = False iif colors = [] iif indicator_colors = None
        patch_artist = colors
        bplot = axi.boxplot(values, labels=labels, whis=[10, 90],
                            showfliers=False, patch_artist=patch_artist)
        if colors:
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.title(ind_name)
