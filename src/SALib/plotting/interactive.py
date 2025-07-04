# -*- coding: utf-8 -*-
"""
This module will contain functions for generating interactive plots using Plotly.

These functions will typically take SALib ResultDict objects or Pandas DataFrames
as input and return Plotly Figure objects.

This module requires Plotly to be installed.
"""

PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    # Plotly not available, functions will raise an error or return a message.
    pass

def _check_plotly_availability():
    """Checks if Plotly is installed and raises an error if not."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive plotting features. "
            "Please install it, e.g., `pip install plotly` or "
            "`pip install salib[plotting_interactive]` (once defined in pyproject.toml)."
        )

import pandas as pd
import numpy as np

# Actual implementation of raincloud_interactive
def raincloud_interactive(
    data,
    param_names=None,
    title=None,
    orientation='v',
    show_box=True,
    show_violin=True,
    show_points='all',
    point_jitter=0.3,
    point_position=-0.5, # Position of points relative to violin/box center
    colors=None, # List of colors for categories, or single color
    violin_kwargs=None,
    box_kwargs=None,
    scatter_kwargs=None,
    layout_kwargs=None
    ):
    """
    Generates an interactive raincloud plot for visualizing distributions.

    A raincloud plot combines a violin plot (density), a box plot (quartiles),
    and individual data points (the "rain"), typically with the density
    on one side and points on the other.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        The data to plot.
        If 2D NumPy array: assumes (observations x parameters/categories).
        If Pandas DataFrame: Assumes wide format (parameters as columns).
                             Support for pre-formatted long DataFrames can be added.
    param_names : list of str, optional
        Names for each parameter/category. If None and data is DataFrame,
        uses column names. If None and data is NumPy array, uses generic names.
    title : str, optional
        Title for the plot.
    orientation : {'v', 'h'}, optional
        Orientation ('v' for vertical, 'h' for horizontal). Default 'v'.
    show_box : bool, optional
        If True, includes a box plot component. Default True.
    show_violin : bool, optional
        If True, includes a violin (density) plot component. Default True.
    show_points : {'all', False}, optional
        If 'all', shows all points as a strip/scatter plot. If False, no points shown.
        Default 'all'. (Note: violin and box traces also have 'points' options
        for outliers, but this controls a dedicated scatter trace for the "rain").
    point_jitter : float, optional
        Amount of jitter for points in the scatter trace. Default 0.3.
    point_position : float, optional
        Position offset for the rain points relative to the center of violin/box.
        E.g., -0.5 for left (vertical) or below (horizontal). Default -0.5.
    colors : list of str, or str, optional
        List of colors for each category, or a single color for all.
        If None, Plotly's default color sequence is used.
    violin_kwargs : dict, optional
        Additional keyword arguments passed to `plotly.graph_objects.Violin`.
    box_kwargs : dict, optional
        Additional keyword arguments passed to `plotly.graph_objects.Box`.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to `plotly.graph_objects.Scatter`.
    layout_kwargs : dict, optional
        Additional keyword arguments passed to `fig.update_layout()`.


    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    _check_plotly_availability()

    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise ValueError("`data` must be a NumPy array or Pandas DataFrame.")

    # Prepare data in long format
    if isinstance(data, np.ndarray):
        if data.ndim == 1: data = data.reshape(-1, 1)
        if data.ndim > 2: raise ValueError("NumPy `data` must be 1D or 2D.")

        _param_names = param_names
        if _param_names is None:
            _param_names = [f"Var {i+1}" for i in range(data.shape[1])]
        elif len(_param_names) != data.shape[1]:
            raise ValueError("Length of `param_names` must match data columns.")

        df_list = [pd.DataFrame({'Category': _param_names[i], 'Value': data[:, i]}) for i in range(data.shape[1])]
        df_long = pd.concat(df_list, ignore_index=True)
    else: # pd.DataFrame (assumed wide)
        _param_names = param_names if param_names is not None else data.columns.tolist()
        df_long = data.melt(value_vars=_param_names, var_name='Category', value_name='Value')

    # Ensure consistent category order
    category_order = _param_names if _param_names is not None else df_long['Category'].unique()

    fig = go.Figure()

    # Default arguments for subplots
    _violin_kwargs = violin_kwargs or {}
    _box_kwargs = box_kwargs or {}
    _scatter_kwargs = scatter_kwargs or {}
    _layout_kwargs = layout_kwargs or {}

    for i, cat_name in enumerate(category_order):
        cat_data = df_long[df_long['Category'] == cat_name]['Value']
        current_color = colors[i % len(colors)] if isinstance(colors, list) else colors

        if orientation == 'v':
            x_cat_violin = [cat_name] * len(cat_data) # For violin/box x-axis
            y_val_violin = cat_data
            x_cat_points = [cat_name] * len(cat_data) # For points x-axis
            y_val_points = cat_data
        else: # orientation == 'h'
            x_val_violin = cat_data
            y_cat_violin = [cat_name] * len(cat_data)
            x_val_points = cat_data
            y_cat_points = [cat_name] * len(cat_data)


        if show_violin:
            fig.add_trace(go.Violin(
                x=x_cat_violin if orientation == 'v' else y_val_violin,
                y=y_val_violin if orientation == 'v' else y_cat_violin,
                name=cat_name,
                side='positive', # Half-violin
                orientation=orientation,
                points=False,
                legendgroup=cat_name,
                showlegend= (i==0), # Show legend only for the first category's violin
                scalegroup=cat_name,
                meanline_visible=False,
                marker_color=current_color,
                **_violin_kwargs
            ))

        if show_box:
            fig.add_trace(go.Box(
                x=x_cat_violin if orientation == 'v' else y_val_violin,
                y=y_val_violin if orientation == 'v' else y_cat_violin,
                name=cat_name,
                boxpoints=False,
                orientation=orientation,
                legendgroup=cat_name,
                showlegend=False, # Box legend item controlled by violin or if violin is hidden
                marker_color=current_color,
                fillcolor='rgba(0,0,0,0)',
                line={'color': current_color if current_color else 'rgba(0,0,0,0.8)'},
                **_box_kwargs
            ))

    # Point display logic using violin/box capabilities
    # This ensures points are part of the primary trace for that category if shown
    point_traces_to_add = []
    if show_points == 'all':
        for i, cat_name in enumerate(category_order):
            cat_data = df_long[df_long['Category'] == cat_name]['Value']
            current_color = colors[i % len(colors)] if isinstance(colors, list) else colors

            if show_violin: # Add points to violin trace
                # Find the violin trace for this category and update it
                for trace in fig.data:
                    if isinstance(trace, go.Violin) and trace.name == cat_name:
                        trace.points = 'all'
                        trace.jitter = point_jitter
                        trace.pointpos = point_position
                        trace.marker.color = current_color # Ensure points match violin color
                        trace.marker.opacity = 0.6 # Make points slightly transparent
                        break
            elif show_box: # No violin, but show box, add points to box trace
                 for trace in fig.data:
                    if isinstance(trace, go.Box) and trace.name == cat_name:
                        trace.boxpoints = 'all'
                        trace.jitter = point_jitter
                        trace.pointpos = point_position # Position relative to box center
                        trace.marker.color = current_color
                        trace.marker.opacity = 0.6
                        break
            else: # Neither violin nor box, add a dedicated scatter trace for points
                # This case might be rare if user wants a "raincloud" but disables both cloud and box
                # For simplicity, if this path is taken, legend might become cluttered.
                # Defaulting to not show legend for these standalone points unless explicitly managed.
                scatter_trace = go.Scatter(
                    x=[cat_name] * len(cat_data) if orientation == 'v' else cat_data,
                    y=cat_data if orientation == 'v' else [cat_name] * len(cat_data),
                    mode='markers',
                    name=cat_name,
                    marker=dict(color=current_color, size=5, opacity=0.6,
                                line=dict(width=0.5, color='DarkSlateGrey')),
                    legendgroup=cat_name,
                    showlegend=(i==0), # Only if it's the very first element plotted
                    **_scatter_kwargs
                )
                if orientation == 'v':
                    scatter_trace.update(xhoverinfo='y', yhoverinfo='skip')
                    if point_jitter > 0:
                         scatter_trace.update(xsrc=df_long[df_long['Category']==cat_name]['Category']) # Required for px-like jitter
                         scatter_trace.update(boxpoints='all', jitter=point_jitter, pointpos=point_position)
                else: # horizontal
                    scatter_trace.update(xhoverinfo='skip', yhoverinfo='x')
                    if point_jitter > 0:
                        scatter_trace.update(ysrc=df_long[df_long['Category']==cat_name]['Category'])
                        scatter_trace.update(boxpoints='all', jitter=point_jitter, pointpos=point_position)
                point_traces_to_add.append(scatter_trace)

    for pt in point_traces_to_add:
        fig.add_trace(pt)

    # Layout updates
    # Ensure legend shows one item per category by default if multiple traces share a legendgroup
    fig.update_layout(title_text=title, showlegend=True)

    if orientation == 'v':
        fig.update_xaxes(categoryorder='array', categoryarray=list(category_order),
                         title_text=_layout_kwargs.pop('xaxis_title', "Parameter"))
        fig.update_yaxes(title_text=_layout_kwargs.pop('yaxis_title', "Value"))
    else: # horizontal
        fig.update_yaxes(categoryorder='array', categoryarray=list(category_order),
                         title_text=_layout_kwargs.pop('yaxis_title', "Parameter"))
        fig.update_xaxes(title_text=_layout_kwargs.pop('xaxis_title', "Value"))

    # Apply other layout kwargs
    fig.update_layout(**_layout_kwargs)

    # Adjust gaps for raincloud aesthetics
    fig.update_layout(violingap=0, boxgap=0,bargap=0) # No gap between elements of same category trace
    fig.update_layout(violingroupgap=0, boxgroupgap=0) # No gap between category groups for violin/box


    return fig


def bar_plot_interactive(
    Si_results,
    index_key,
    conf_key=None,
    param_names=None,
    title=None,
    orientation='v',
    sort_by_value=None, # None, 'ascending', 'descending'
    color_discrete_map=None,
    bar_kwargs=None,
    layout_kwargs=None
    ):
    """
    Generates an interactive bar chart for sensitivity indices.

    Parameters
    ----------
    Si_results : ResultDict or pd.DataFrame
        Sensitivity analysis results. If ResultDict, it's expected to have
        `names` and the specified `index_key` (and `conf_key` if used).
        If DataFrame, it should have columns for parameters (index or column),
        the `index_key`, and optionally `conf_key`.
    index_key : str
        Key or column name for the sensitivity index values to plot.
    conf_key : str, optional
        Key or column name for the confidence interval values. If provided,
        these are displayed as error bars.
    param_names : list of str, optional
        Names for parameters/categories. If None, inferred from `Si_results`.
    title : str, optional
        Plot title.
    orientation : {'v', 'h'}, optional
        Orientation ('v' for vertical bars, 'h' for horizontal). Default 'v'.
    sort_by_value : {None, 'ascending', 'descending'}, optional
        If not None, sorts bars by index value. Default None (uses input order).
    color_discrete_map : dict, optional
        A dictionary mapping parameter names to colors. E.g., {'Param1': 'blue'}.
    bar_kwargs : dict, optional
        Additional keyword arguments passed to `plotly.express.bar()`.
    layout_kwargs : dict, optional
        Additional keyword arguments passed to `fig.update_layout()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    _check_plotly_availability()

    if not isinstance(Si_results, (ResultDict, pd.DataFrame)):
        raise ValueError("`Si_results` must be a SALib ResultDict or Pandas DataFrame.")

    _bar_kwargs = bar_kwargs or {}
    _layout_kwargs = layout_kwargs or {}

    # Prepare DataFrame for Plotly Express
    if isinstance(Si_results, ResultDict):
        if param_names is None:
            _param_names = Si_results.get('names')
            if _param_names is None:
                raise ValueError("`param_names` must be provided or available in ResultDict['names'].")
        else:
            _param_names = param_names

        values = Si_results[index_key]
        df_data = {'Parameter': _param_names, index_key: values}

        error_bar_col = None
        if conf_key:
            if conf_key not in Si_results:
                raise ValueError(f"Confidence interval key '{conf_key}' not found in Si_results.")
            df_data[conf_key] = Si_results[conf_key]
            error_bar_col = conf_key

        plot_df = pd.DataFrame(df_data)

    elif isinstance(Si_results, pd.DataFrame):
        plot_df = Si_results.copy()
        # Try to infer param_names if not provided.
        # If DataFrame index is meaningful (e.g. parameter names), use it.
        # Otherwise, assume a column named 'Parameter' or 'names', or use first string column.
        if param_names is None:
            if plot_df.index.name is not None and isinstance(plot_df.index.name, str) and plot_df.index.name.lower() not in ['rangeindex', 'int64index']:
                 _param_names_col = plot_df.index.name
                 plot_df[_param_names_col] = plot_df.index # Make index a column
            elif 'Parameter' in plot_df.columns:
                _param_names_col = 'Parameter'
            elif 'names' in plot_df.columns:
                _param_names_col = 'names'
            else: # Try to find first string column to use as category
                _param_names_col = plot_df.select_dtypes(include=['object', 'string']).columns.tolist()
                if not _param_names_col:
                    raise ValueError("Could not infer parameter names column from DataFrame. Please specify `param_names` or ensure a suitable column exists.")
                _param_names_col = _param_names_col[0]
        else: # If param_names list is given, ensure a column exists or create one
            if 'Parameter' not in plot_df.columns and not (plot_df.index.name == 'Parameter' or (param_names and len(param_names) == len(plot_df))):
                 # This case is tricky if param_names list is given but no obvious column matches
                 # For now, assume if DataFrame, param_names refers to an existing column or index
                 raise ValueError("If `param_names` is used with DataFrame, it's for selecting a specific column to be used as parameter labels, or it's ignored if structure is obvious.")
            _param_names_col = 'Parameter' # Standardize for px call

        if index_key not in plot_df.columns:
            raise ValueError(f"Index key '{index_key}' not found in DataFrame columns.")

        error_bar_col = None
        if conf_key:
            if conf_key not in plot_df.columns:
                raise ValueError(f"Confidence interval key '{conf_key}' not found in DataFrame columns.")
            error_bar_col = conf_key

    # Sorting
    if sort_by_value:
        plot_df = plot_df.sort_values(by=index_key, ascending=(sort_by_value == 'ascending'))

    # Determine x, y, and error bar parameters based on orientation
    if orientation == 'v':
        x_arg = _param_names_col
        y_arg = index_key
        error_arg = f"error_y"
    else: # orientation == 'h'
        x_arg = index_key
        y_arg = _param_names_col
        error_arg = f"error_x"

    # Add error bar dict to bar_kwargs if conf_key is present
    if error_bar_col:
        _bar_kwargs[error_arg] = error_bar_col
        # _bar_kwargs[error_arg + "_symmetric"] = False # If CIs are [val-low, val+high] vs just width
        # Assuming conf_key provides symmetric error value for now.

    fig = px.bar(plot_df,
                 x=x_arg,
                 y=y_arg,
                 title=title,
                 orientation=orientation,
                 color=_param_names_col if color_discrete_map else None, # Color bars by parameter name if map provided
                 color_discrete_map=color_discrete_map,
                 **_bar_kwargs)

    # Update layout
    _final_layout_kwargs = {}
    if orientation == 'v':
        _final_layout_kwargs['xaxis_title'] = _layout_kwargs.pop('xaxis_title', "Parameter")
        _final_layout_kwargs['yaxis_title'] = _layout_kwargs.pop('yaxis_title', index_key)
        # Ensure categorical x-axis order matches sorted data if sorting applied
        if sort_by_value:
             _final_layout_kwargs['xaxis'] = {'categoryorder':'array', 'categoryarray': plot_df[_param_names_col].tolist()}
    else: # orientation == 'h'
        _final_layout_kwargs['yaxis_title'] = _layout_kwargs.pop('yaxis_title', "Parameter")
        _final_layout_kwargs['xaxis_title'] = _layout_kwargs.pop('xaxis_title', index_key)
        if sort_by_value:
            _final_layout_kwargs['yaxis'] = {'categoryorder':'array', 'categoryarray': plot_df[_param_names_col].tolist()}

    fig.update_layout(**_final_layout_kwargs)
    fig.update_layout(**_layout_kwargs) # Apply remaining user layout kwargs

    return fig


# def scatter_plot_interactive(Si, ...):
#     _check_plotly_availability()
#     # ... implementation ...

# def heatmap_interactive(Si, ...):
#     _check_plotly_availability()
#     # ... implementation ...
