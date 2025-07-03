# -*- coding: utf-8 -*-
"""
Tests for interactive plotting functions (using Plotly).

These tests will primarily check:
- If plotting functions run without error with valid inputs.
- If they handle missing Plotly gracefully (if PLOTLY_AVAILABLE is False).
- Basic properties of the returned Plotly Figure objects (e.g., not None).
Detailed visual verification and interactivity testing is typically done manually.
"""
import pytest
import numpy as np
import pandas as pd

from SALib.util import ResultDict
# Assuming the new module will be SALib.plotting.interactive
# We'll need to try importing its functions and handle PLOTLY_AVAILABLE status.

# Example:
# from SALib.plotting.interactive import bar_plot_interactive, PLOTLY_AVAILABLE
#
# @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
# class TestInteractiveBarPlot:
#     def test_simple_bar_plot(self):
#         # Setup dummy ResultDict or DataFrame
#         problem = {'num_vars': 3, 'names': ['x1', 'x2', 'x3']}
#         Si = ResultDict({
#             'S1': np.array([0.1, 0.2, 0.3]),
#             'S1_conf': np.array([0.01, 0.02, 0.03]),
#             'names': problem['names']
#         })
#         Si.problem = problem # Attach problem spec for context if needed by plot
#
#         fig = bar_plot_interactive(Si, index_key='S1', conf_key='S1_conf')
#         assert fig is not None
#         # Add more checks if possible, e.g., number of traces, layout properties
from SALib.plotting.interactive import raincloud_interactive, PLOTLY_AVAILABLE


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed, skipping interactive plot tests.")
class TestRaincloudInteractive:
    def test_raincloud_numpy_2d_defaults(self):
        """Test with 2D NumPy array and default settings."""
        data = np.random.rand(100, 3)
        fig = raincloud_interactive(data)
        assert fig is not None
        assert len(fig.data) > 0 # Should have traces for violin, box, points per param
        # Default is 3 traces per category if all shown (violin, box, points on violin)
        # Current implementation: violin, box. Points are part of violin. So 2 traces per category if all shown.
        # If show_violin=T, show_box=T, show_points='all' -> violin (with points) + box = 2 traces per category.
        # Number of categories = 3. So 3 * 2 = 6 traces.
        # Let's check based on what's enabled.
        num_traces = 0
        if True: num_traces += data.shape[1] # Violins (with points)
        if True: num_traces += data.shape[1] # Boxes
        # The points are now part of violin/box, not separate traces.
        # So, if show_violin and show_box, we expect D violins + D boxes = 2D traces.
        # If only violin (with points), D traces. If only box (with points), D traces.

        expected_traces = 0
        # Current raincloud_interactive logic:
        # - if show_violin, adds D violin traces. These will show points if show_points='all'.
        # - if show_box, adds D box traces. These will show points if show_points='all' AND NOT show_violin.
        if True: # show_violin is default True
            expected_traces += data.shape[1]
        if True: # show_box is default True
            expected_traces += data.shape[1]
        assert len(fig.data) == expected_traces


    def test_raincloud_numpy_1d(self):
        """Test with 1D NumPy array."""
        data = np.random.rand(100)
        fig = raincloud_interactive(data, param_names=['Single Param'])
        assert fig is not None
        assert len(fig.data) > 0
        expected_traces = 0
        if True: expected_traces += 1
        if True: expected_traces += 1
        assert len(fig.data) == expected_traces


    def test_raincloud_pandas_df_wide(self):
        """Test with Pandas DataFrame in wide format."""
        df = pd.DataFrame(np.random.rand(100, 2), columns=['A', 'B'])
        fig = raincloud_interactive(df)
        assert fig is not None
        assert len(fig.data) > 0
        expected_traces = 0
        if True: expected_traces += df.shape[1]
        if True: expected_traces += df.shape[1]
        assert len(fig.data) == expected_traces

    def test_raincloud_options_orientation_title_colors(self):
        """Test various options like orientation, title, colors."""
        data = np.random.rand(50, 2)
        param_names = ['Var1', 'Var2']
        title_test = "Test Raincloud Plot"
        colors_test = ['rgba(255,0,0,0.6)', 'rgba(0,0,255,0.6)']

        fig = raincloud_interactive(data,
                                    param_names=param_names,
                                    title=title_test,
                                    orientation='h',
                                    colors=colors_test)
        assert fig is not None
        assert fig.layout.title.text == title_test
        assert fig.data[0].orientation == 'h' # First violin
        assert fig.data[0].marker.color == colors_test[0] # Color of first violin
        assert fig.data[1].marker.color == colors_test[1] # Color of second violin (assuming 2 violins)


    def test_raincloud_hide_elements(self):
        """Test hiding violin, box, or points."""
        data = np.random.rand(50, 1)
        fig_no_box = raincloud_interactive(data, show_box=False)
        # Expect 1 trace (violin with points)
        assert len(fig_no_box.data) == 1
        assert isinstance(fig_no_box.data[0], go.Violin)

        fig_no_violin = raincloud_interactive(data, show_violin=False)
        # Expect 1 trace (box with points)
        assert len(fig_no_violin.data) == 1
        assert isinstance(fig_no_violin.data[0], go.Box)

        fig_no_points_on_violin = raincloud_interactive(data, show_points=False, show_box=False)
        assert len(fig_no_points_on_violin.data) == 1 # Just violin
        assert fig_no_points_on_violin.data[0].points == False

        fig_no_points_on_box = raincloud_interactive(data, show_points=False, show_violin=False)
        assert len(fig_no_points_on_box.data) == 1 # Just box
        assert fig_no_points_on_box.data[0].boxpoints == False


    def test_raincloud_invalid_data(self):
        """Test error handling for invalid data types."""
        with raises(ValueError, match="`data` must be a NumPy array or Pandas DataFrame."):
            raincloud_interactive([1, 2, 3]) # list input

        with raises(ValueError, match="NumPy `data` must be 1D or 2D."):
            raincloud_interactive(np.random.rand(10,2,3)) # 3D numpy

    def test_raincloud_param_names_mismatch(self):
        """Test error if param_names length mismatches data columns."""
        data = np.random.rand(50, 3)
        with raises(ValueError, match="Length of `param_names` must match data columns."):
            raincloud_interactive(data, param_names=['P1', 'P2'])


# Example of how to test if Plotly is NOT available (requires mocking or specific test setup)
# @patch('SALib.plotting.interactive.PLOTLY_AVAILABLE', False)
# def test_plotly_not_available(mock_plotly_unavailable):
#     with pytest.raises(ImportError, match="Plotly is required"):
#         raincloud_interactive(np.random.rand(10,1))
# This type of test is more involved due to module-level PLOTLY_AVAILABLE check.
from SALib.plotting.interactive import bar_plot_interactive # Add import

# For now, we rely on @pytest.mark.skipif for when Plotly is genuinely not there.


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed, skipping interactive plot tests.")
class TestInteractiveBarPlot:
    @pytest.fixture
    def sample_si_dict(self):
        """Return a sample ResultDict for bar plot tests."""
        problem = {'num_vars': 3, 'names': ['X1', 'X2', 'X3']}
        si = ResultDict({
            'S1': np.array([0.3, 0.1, 0.2]),
            'S1_conf': np.array([0.05, 0.02, 0.03]),
            'ST': np.array([0.5, 0.2, 0.4]),
            'ST_conf': np.array([0.06, 0.03, 0.04]),
            'names': problem['names']
        })
        si.problem = problem # Typically not needed by plot func directly, but good practice
        return si

    @pytest.fixture
    def sample_df(self, sample_si_dict):
        """Return a sample DataFrame for bar plot tests."""
        # Create a DataFrame similar to what Si.to_df() might produce for one index
        df = pd.DataFrame({
            'Parameter': sample_si_dict['names'],
            'S1': sample_si_dict['S1'],
            'S1_conf': sample_si_dict['S1_conf']
        }).set_index('Parameter')
        return df

    def test_bar_plot_resultdict_defaults(self, sample_si_dict):
        """Test with ResultDict and default settings."""
        fig = bar_plot_interactive(sample_si_dict, index_key='S1')
        assert fig is not None
        assert len(fig.data) == 1 # One bar trace
        assert fig.data[0].type == 'bar'
        assert len(fig.data[0].x) == len(sample_si_dict['names']) # Check number of bars

    def test_bar_plot_dataframe_defaults(self, sample_df):
        """Test with Pandas DataFrame and default settings."""
        # For DataFrame, need to specify param_names via index or a column.
        # Current bar_plot_interactive infers from index if name is not standard.
        # Let's test with index.
        df_to_plot = sample_df.reset_index() # Make 'Parameter' a column
        fig = bar_plot_interactive(df_to_plot, index_key='S1') # param_names_col inferred as 'Parameter'
        assert fig is not None
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == len(sample_df)

    def test_bar_plot_with_conf_intervals(self, sample_si_dict):
        """Test with confidence intervals."""
        fig = bar_plot_interactive(sample_si_dict, index_key='S1', conf_key='S1_conf')
        assert fig is not None
        assert fig.data[0].error_y is not None # Check if error bars are configured
        assert fig.data[0].error_y.type == 'data'

    def test_bar_plot_horizontal(self, sample_si_dict):
        """Test horizontal orientation."""
        fig = bar_plot_interactive(sample_si_dict, index_key='S1', orientation='h')
        assert fig is not None
        assert fig.data[0].orientation == 'h'

    def test_bar_plot_sorting(self, sample_si_dict):
        """Test sorting by value."""
        # Ascending
        fig_asc = bar_plot_interactive(sample_si_dict, index_key='S1', sort_by_value='ascending')
        assert fig_asc is not None
        # Expected order of names for S1 [0.3, 0.1, 0.2] ascending: X2, X3, X1
        expected_order_asc = ['X2', 'X3', 'X1']
        assert list(fig_asc.layout.xaxis.categoryarray) == expected_order_asc

        # Descending
        fig_desc = bar_plot_interactive(sample_si_dict, index_key='S1', sort_by_value='descending')
        assert fig_desc is not None
        expected_order_desc = ['X1', 'X3', 'X2']
        assert list(fig_desc.layout.xaxis.categoryarray) == expected_order_desc

    def test_bar_plot_title_and_colors(self, sample_si_dict):
        """Test title and color_discrete_map options."""
        title_test = "Test Bar Plot"
        colors_test = {'X1': 'red', 'X2': 'green', 'X3': 'blue'}
        fig = bar_plot_interactive(sample_si_dict, index_key='S1', title=title_test, color_discrete_map=colors_test)
        assert fig is not None
        assert fig.layout.title.text == title_test
        # Check if colors are applied (Plotly Express handles mapping color to traces)
        # This check is indirect: if color_discrete_map is used, px.bar uses the 'color' arg.
        # The actual marker colors would be in fig.data[0].marker.color, but could be an array.
        # For simplicity, just check that it runs and uses the color argument.
        assert fig.data[0].marker.color is not None # Will be an array of colors if map is used

    def test_bar_plot_invalid_input_type(self):
        with raises(ValueError, match="`Si_results` must be a SALib ResultDict or Pandas DataFrame."):
            bar_plot_interactive([1,2,3], index_key='S1')

    def test_bar_plot_missing_keys(self, sample_si_dict):
        with raises(KeyError): # ResultDict raises KeyError
            bar_plot_interactive(sample_si_dict, index_key='NonExistentKey')

        with raises(ValueError, match="Confidence interval key 'NonExistentConfKey' not found"):
            bar_plot_interactive(sample_si_dict, index_key='S1', conf_key='NonExistentConfKey')

    def test_bar_plot_missing_param_names_in_resultdict(self):
        si_no_names = ResultDict({'S1': np.array([0.1,0.2])}) # No 'names' key
        with raises(ValueError, match="`param_names` must be provided or available in ResultDict['names']"):
            bar_plot_interactive(si_no_names, index_key='S1')
