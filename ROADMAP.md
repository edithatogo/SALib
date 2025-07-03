# SALib Roadmap

This document outlines potential future directions and features for the SALib library.
The order does not strictly imply priority, as that will be determined by community needs, contributions, and maintainer capacity. This is a living document and subject to change.

## Guiding Principles for Future Development

*   **Ease of Use:** Continue to make sensitivity analysis accessible to a wide range of users.
*   **Comprehensiveness:** Offer a robust set of established and emerging SA methods.
*   **Performance:** Ensure the library is efficient for increasingly complex models and analyses.
*   **Interoperability:** Facilitate integration with other scientific Python libraries and modeling workflows.
*   **Community-Driven:** Encourage contributions and incorporate feedback from the user community.

## Potential Feature Areas

Here are some areas we are considering for future enhancements. Contributions and discussions are welcome!

### 1. Enhanced Computational Efficiency & Scalability

*   **Status:** Ongoing (e.g., recent parallelization of bootstrap routines).
*   **Next Steps:**
    *   [ ] **Broader Parallelization:** Investigate parallelization opportunities beyond bootstrapping in core SA algorithms (e.g., in sampling, matrix operations for certain methods).
    *   [ ] **Distributed Computing:** Explore deeper integration with libraries like Dask or Ray for analyses that exceed single-machine capabilities. This could build upon the existing `pathos` optional dependency.
    *   [ ] **Performance Profiling:** Add tools or documentation to help users understand and optimize the performance of their SA studies.

### 2. Advanced Visualizations

*   **Goal:** Provide more interactive and insightful ways to explore SA results.
*   **Potential Next Steps:**
    *   [ ] **Interactive Plots:** Integrate with libraries like Plotly, Bokeh, or Altair to create interactive versions of existing plots (e.g., bar plots, scatter plots of indices).
    *   [ ] **New Visualization Types:** Develop visualizations for higher-order interactions, grouped analyses, or time-dependent outputs.
    *   [ ] **Dashboarding Capabilities:** Explore options for creating simple dashboards to present comprehensive SA results.

### 3. Surrogate Model (Metamodeling) Integration

*   **Goal:** Make it easier to perform SA on computationally expensive models by using surrogate models.
*   **Potential Next Steps:**
    *   [ ] **Documentation & Examples:** Provide clear examples and best-practice guides for using SALib with common surrogate modeling libraries (e.g., scikit-learn for GPs, PySOT, UQPy).
    *   [ ] **Utility Functions:** Consider helper functions for common tasks like fitting a surrogate to model outputs and then running SA on the surrogate.
    *   [ ] **Surrogate-Specific SA:** Investigate SA methods specifically designed for or optimized for use with certain types of surrogate models.

### 4. Expanded Methodological Support

*   **Goal:** Continue to incorporate valuable SA methods.
*   **Potential Next Steps (Community input needed for prioritization):**
    *   [ ] **Methods for Correlated Inputs:** More explicit support or dedicated methods for problems with correlated input parameters.
    *   [ ] **Methods for Time-Dependent or Functional Outputs:** Techniques to analyze sensitivity when model outputs are time series or functions.
    *   [ ] **Goal-Oriented SA:** Methods that focus on the sensitivity of specific model outcomes or decisions.
    *   [ ] **Factor Fixing/Screening:** More advanced screening methods to efficiently identify non-influential parameters.

### 5. Improved User Experience & Workflow

*   **Goal:** Streamline the process of setting up and running SA studies.
*   **Potential Next Steps:**
    *   [ ] **Enhanced `ProblemSpec`:** Further develop the `ProblemSpec` interface for more complex problem definitions or result management.
    *   [ ] **Model Interface Adapters:** Consider adapters or wrappers for common modeling frameworks to simplify the evaluation loop.
    *   [ ] **Clearer Guidance on Method Selection:** Improve documentation or add tools to help users choose the most appropriate SA method for their problem.

### 6. Documentation and Community

*   **Goal:** Foster a vibrant and well-supported user and developer community.
*   **Next Steps:**
    *   [ ] **Tutorials & Case Studies:** Develop more in-depth tutorials and real-world case studies.
    *   [ ] **Contribution Guidelines:** Continuously refine guidelines for contributors.
    *   [ ] **API Documentation:** Ensure API documentation is comprehensive and up-to-date.

## How to Contribute

We welcome contributions! Please see `CONTRIBUTING.md` for more details on how to get involved. You can also open an issue on GitHub to discuss potential features or report bugs.
