.. _interactive-plotting:

Interactive Plotting (Experimental)
===================================

SALib offers experimental support for generating interactive plots using the
Plotly library. These plots can enhance the exploration and presentation of
sensitivity analysis results.

**Note:** This feature is experimental and requires Plotly to be installed.

Installation
------------

To use the interactive plotting features, you need to install Plotly.
You can install it along with SALib using the `plotting_interactive` extra:

.. code-block:: bash

   pip install salib[plotting_interactive]

Or, you can install Plotly directly:

.. code-block:: bash

   pip install plotly>=5.0

If Plotly is not installed, calling these functions will raise an `ImportError`.

Available Interactive Plots
-------------------------

All interactive plotting functions are available under the `SALib.plotting.interactive` module.
They typically return a `plotly.graph_objects.Figure` object, which can be shown
in Jupyter notebooks, Dash apps, or exported to HTML.

.. contents::
   :local:
   :depth: 1

Raincloud Plots
~~~~~~~~~~~~~~~

Raincloud plots provide a rich visualization of data distributions, combining
elements of a violin plot (density estimate), a box plot (quartiles), and
individual data points (the "rain"). They are particularly useful for:

*   Visualizing the distribution of bootstrapped sensitivity indices.
*   Displaying the distributions of Morris elementary effects for each parameter.

**Function:** `SALib.plotting.interactive.raincloud_interactive`

**Key Parameters:**

*   `data`: A 2D NumPy array (observations x parameters) or a Pandas DataFrame (wide format) containing the data for which distributions are to be plotted.
*   `param_names`: List of parameter names for labeling.
*   `orientation`: 'v' (vertical) or 'h' (horizontal).
*   `show_violin`, `show_box`, `show_points`: Booleans to control plot components.
*   `point_jitter`, `point_position`: Control appearance of individual data points.
*   `colors`: List of colors or single color.
*   `violin_kwargs`, `box_kwargs`, `layout_kwargs`: Dictionaries for passing custom arguments to Plotly.

**Example Usage:**

.. code-block:: python

    from SALib.analyze import sobol
    from SALib.sample import saltelli
    from SALib.test_functions import Ishigami
    from SALib.plotting.interactive import raincloud_interactive
    import numpy as np

    # Assume `problem` is defined (e.g., for Ishigami)
    # problem = {
    #     'num_vars': 3,
    #     'names': ['x1', 'x2', 'x3'],
    #     'bounds': [[-np.pi, np.pi]]*3
    # }

    # Generate samples and run model (example)
    # X = saltelli.sample(problem, 1024, calc_second_order=False)
    # Y = Ishigami.evaluate(X)

    # Perform Sobol analysis with bootstrapping to get S1 distributions
    # Si = sobol.analyze(problem, Y, calc_second_order=False,
    #                    num_resamples=1000, keep_resamples=True)

    # Assuming Si['S1_conf_all'] contains bootstrapped S1 values (num_resamples x num_vars)
    # This part is conceptual as `keep_resamples` for Sobol currently stores CIs, not all resamples.
    # For demonstration, let's create dummy bootstrap data:
    num_vars = 3
    num_resamples_for_plot = 200
    dummy_s1_bootstrap_data = np.random.rand(num_resamples_for_plot, num_vars)
    dummy_param_names = [f'Param {i+1}' for i in range(num_vars)]

    # Create the raincloud plot
    # try:
    #     fig_raincloud = raincloud_interactive(
    #         dummy_s1_bootstrap_data,
    #         param_names=dummy_param_names,
    #         title="Distribution of S1 Indices (Bootstrapped)",
    #         orientation='h'
    #     )
    #     fig_raincloud.show() # In a Jupyter environment
    # except ImportError:
    #     print("Plotly not installed, skipping raincloud plot example.")

*(The plot would show horizontal half-violins with box plots and jittered points for each parameter's S1 index distribution, allowing comparison of their spread and central tendency.)*


Interactive Bar Charts
~~~~~~~~~~~~~~~~~~~~~~

Enhances the standard bar chart for sensitivity indices by adding interactivity such as tooltips (showing exact values and confidence intervals) and optional sorting.

**Function:** `SALib.plotting.interactive.bar_plot_interactive`

**Key Parameters:**

*   `Si_results`: A `ResultDict` from a SALib analysis or a Pandas DataFrame.
*   `index_key`: The key for the sensitivity index values (e.g., 'S1', 'ST').
*   `conf_key`: Optional key for confidence interval values (to show as error bars).
*   `param_names`: List of parameter names.
*   `orientation`: 'v' or 'h'.
*   `sort_by_value`: `None`, 'ascending', or 'descending'.
*   `color_discrete_map`: Dictionary to map parameter names to colors.
*   `bar_kwargs`, `layout_kwargs`: Dictionaries for Plotly customization.

**Example Usage:**

.. code-block:: python

    from SALib.analyze import sobol
    # from SALib.sample import saltelli # Assuming already imported
    # from SALib.test_functions import Ishigami # Assuming already imported
    from SALib.plotting.interactive import bar_plot_interactive
    # import numpy as np # Assuming already imported

    # problem = {
    #     'num_vars': 3,
    #     'names': ['x1', 'x2', 'x3'],
    #     'bounds': [[-np.pi, np.pi]]*3
    # }
    # X = saltelli.sample(problem, 1024)
    # Y = Ishigami.evaluate(X)
    # Si_sobol = sobol.analyze(problem, Y, calc_second_order=True)

    # For demonstration, create a dummy Si_sobol ResultDict
    # from SALib.util import ResultDict
    # Si_sobol = ResultDict({
    #    'S1': np.array([0.3, 0.4, 0.05]), 'S1_conf': np.array([0.05, 0.06, 0.01]),
    #    'ST': np.array([0.5, 0.45, 0.2]), 'ST_conf': np.array([0.07, 0.06, 0.03]),
    #    'S2': np.array([[np.nan, 0.1, 0.2],[np.nan, np.nan, 0.05],[np.nan, np.nan, np.nan]]),
    #    'S2_conf': np.array([[np.nan,0.02,0.03],[np.nan,np.nan,0.01],[np.nan,np.nan,np.nan]]),
    #    'names': problem['names']
    # })


    # Create an interactive bar plot for S1 indices
    # try:
    #     fig_bar_s1 = bar_plot_interactive(
    #         Si_sobol,
    #         index_key='S1',
    #         conf_key='S1_conf',
    #         title='Interactive First-Order Sobol Indices (S1)',
    #         sort_by_value='descending'
    #     )
    #     fig_bar_s1.show()
    # except ImportError:
    #     print("Plotly not installed, skipping interactive bar plot example.")

*(The plot would show bars for each S1 index, sorted in descending order, with error bars for confidence intervals. Hovering over bars would show details.)*

Further Plot Types
------------------
Additional interactive plot types, such as interactive scatter plots (e.g., ST vs S1) and interactive heatmaps (for S2 indices), are planned for future versions.
Users familiar with Plotly can also use the `ResultDict` (often by converting to DataFrame with `.to_df()`) to create custom interactive visualizations.
