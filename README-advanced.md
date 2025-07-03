## SALib: Advanced options

### Parameter files

In the parameter file, lines beginning with `#` will be treated as comments and ignored.
```
# name lower_bound upper_bound
P1 0.0 1.0
P2 0.0 5.0
P3 0.0 5.0
...etc.
```
Parameter files can also be comma-delimited if your parameter names or group names contain spaces. This should be detected automatically.

### Command-line interface

**Generate samples** (the `-p` flag is the parameter file)
```
salib sample saltelli \
     -n 1024 \
     -p ./src/SALib/test_functions/params/Ishigami.txt \
     -o model_input.txt
```

**Run the model** this will usually be a user-defined model, maybe even in another language. Just save the outputs.

**Run the analysis**
```
salib analyze sobol \
     -p ./src/SALib/test_functions/params/Ishigami.txt \
     -Y model_output.txt \
     -c 0
```

This will print indices and confidence intervals to the command line. You can redirect to a file using the `>` operator.

### Parallel Computation
The Sobol method supports parallel calculation of sensitivity indices.
```python
Si_sobol = sobol.analyze(problem, Y, calc_second_order=True, conf_level=0.95,
                         print_to_console=False, parallel=True, n_processors=4)
```

Additionally, the Morris, FAST, Delta Moment-Independent (Delta-MIM), and Derivative-based Global Sensitivity Measure (DGSM) methods support parallel computation for their bootstrap/resampling procedures (typically used for generating confidence intervals). This can provide a speed-up when `num_resamples` is large.

To use parallel computation for these methods, pass the `parallel=True` and `n_processors` arguments to their respective `analyze` functions:

```python
# Example for Morris
Si_morris = morris.analyze(problem, X, Y, num_resamples=1000, conf_level=0.95,
                           print_to_console=False, parallel=True, n_processors=4)

# Example for FAST
Si_fast = fast.analyze(problem, Y, M=4, num_resamples=1000, conf_level=0.95,
                       print_to_console=False, parallel=True, n_processors=4)

# Example for Delta
Si_delta = delta.analyze(problem, X, Y, num_resamples=1000, conf_level=0.95,
                         print_to_console=False, parallel=True, n_processors=4)

# Example for DGSM
Si_dgsm = dgsm.analyze(problem, X, Y, num_resamples=1000, conf_level=0.95,
                       print_to_console=False, parallel=True, n_processors=4)
```

The `n_processors` argument specifies the number of CPU cores to use. If set to `None` (the default when `parallel=True` but `n_processors` is not given), it will typically use all available cores. The `multiprocess` library is used for parallel execution.

For an explanation of all command line options for each method, [see the examples here](https://github.com/SALib/SALib/tree/main/examples).


### Groups of variables (Sobol and Morris methods only)
It is sometimes useful to perform sensitivity analysis on groups of input variables to reduce the number of model runs required, when variables belong to the same component of a model, or there is some reason to believe that they should behave similarly.

Groups can be specified in two ways for the Sobol and Morris methods. First, as a fourth column in the parameter file:
```
# name lower_bound upper_bound group_name
P1 0.0 1.0 Group_1
P2 0.0 5.0 Group_2
P3 0.0 5.0 Group_2
...etc.
```

Or in the `problem` dictionary:
```python
problem = {
  'groups': ['Group_1', 'Group_2', 'Group_2'],
  'names': ['x1', 'x2', 'x3'],
  'num_vars': 3,
  'bounds': [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]]
}
```

`groups` is a list of strings specifying the group name to which each variable belongs. The rest of the code stays the same:

```python
param_values = saltelli.sample(problem, 1024)
Y = Ishigami.evaluate(param_values)
Si = sobol.analyze(problem, Y, print_to_console=True)
```

But the output is printed by group:
```
Group S1 S1_conf ST ST_conf
Group_1 0.307834 0.066424 0.559577 0.082978
Group_2 0.444052 0.080255 0.667258 0.060871

Group_1 Group_2 S2 S2_conf
Group_1 Group_2 0.242964 0.124229
```

The output can then be converted to a Pandas DataFrame for further analysis.

```python
total_Si, first_Si, second_Si = Si.to_df()
```

### Defining Correlated Inputs

SALib supports the generation of samples with a defined rank correlation structure through Latin Hypercube Sampling (LHS). This is useful when input parameters to a model are known to be correlated.

To specify correlations, include a `corr_matrix` key in your problem definition. The value should be a NumPy array representing the desired rank correlation matrix (Spearman's rho).

**Example Problem Definition with `corr_matrix`:**
```python
import numpy as np

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[0.0, 1.0], [-1.0, 1.0], [10.0, 20.0]],
    # Define desired rank correlation matrix (Spearman's rho)
    'corr_matrix': np.array([[1.0, 0.7, 0.3],
                             [0.7, 1.0, -0.5],
                             [0.3, -0.5, 1.0]]),
    # Optionally, define marginal distributions for each parameter
    'dists': ['unif', 'norm', 'unif'] # x1: U(0,1), x2: N(0,1) (bounds become loc,scale), x3: U(10,20)
}

# If 'dists' for 'norm' is specified, the corresponding 'bounds' entry
# should be [mean, std_dev]. For 'unif', it's [lower_bound, upper_bound].
# If 'dists' is not specified, all parameters are assumed to be uniform
# according to their 'bounds'.
```

**Validation:**
The `corr_matrix` must be:
- A square NumPy array with dimensions equal to `num_vars`.
- Symmetric.
- Have diagonal elements equal to 1.
- Have off-diagonal elements between -1 and 1.
- Positive semi-definite.
These properties are validated when the problem dictionary is used (e.g., by `ProblemSpec` or directly by samplers).

**Sampling with Correlated Inputs:**
Currently, `SALib.sample.latin.sample` (and its `ProblemSpec.sample_latin()` counterpart) can generate samples respecting this `corr_matrix`. It uses an Iman and Conover (1982) style procedure to induce the specified rank correlation.
```python
from SALib.sample import latin
# Assuming 'problem' is defined as above
param_values_correlated = latin.sample(problem, N=1024, seed=101)
```

**Important Considerations for Analysis:**
When input parameters are correlated:
- **Standard sensitivity indices** (e.g., Sobol, FAST, Morris, DGSM) can be misleading if their standard interpretations are used, as these methods often assume input independence. SALib will issue warnings when these analysis methods are used with a problem spec containing a `corr_matrix`.
- **Moment-independent methods** like the Delta Moment-Independent Measure (`delta.analyze`) are generally more robust to input correlations, but interpretation still requires caution.
- Always carefully consider the implications of correlations on your sensitivity analysis results. Specialized SA methods for correlated inputs exist in literature but are not yet all implemented in SALib.

Refer to the documentation of individual analysis functions for specific warnings and guidance.


### Generating alternate distributions

In the [Quick Start](https://github.com/SALib/SALib/tree/main/README.rst) we
generate a uniform sample of parameter space.

```python
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

problem = {
     'num_vars': 3,
     'names': ['x1', 'x2', 'x3'],
     'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]]
}

param_values = saltelli.sample(problem, 1024)
```

SALib is also capable of generating alternate sampling distributions by
specifying a `dist` entry in the `problem` specification.

As implied in the basic example, a uniform distribution is the default.

When an entry for `dist` is not 'unif', the `bounds` entry does not indicate
parameter bounds but sample-specific metadata.

`bounds` definitions for available distributions:

* unif: uniform distribution
    e.g. :code:`[-np.pi, np.pi]` defines the lower and upper bounds
* triang: triangular with lower and upper bounds, as well as
     location of peak
     The location of peak is in percentage of width
     e.g. :code:`[1.0, 3.0, 0.5]` indicates 1.0 to 3.0 with a peak at 2.0

     A soon-to-be deprecated two-value format assumes the lower bound to be 0
     e.g. :code:`[3, 0.5]` assumes 0 to 3, with a peak at 1.5
* norm: normal distribution with mean and standard deviation
* lognorm: lognormal with ln-space mean and standard deviation

An example specification is shown below:

```python
problem = {
     'names': ['x1', 'x2', 'x3'],
     'num_vars': 3,
     'bounds': [[-np.pi, np.pi], [1.0, 0.2], [3, 0.5]],
     'groups': ['G1', 'G2', 'G1'],
     'dists': ['unif', 'lognorm', 'triang']
}
```
