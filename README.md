# torch-graph
Doing it for fun. Also whant to understand better tensorflow core and [dataflow](https://en.wikipedia.org/wiki/Dataflow_programming) paradigm impementations.
Draft of computational graph for pytorch. Intended to be a keras backend in the future.

# Issues & targets:
* How to assign new values to existed torch tensor from `python` without re-allocating it?
* How to use `grad` module of pytorch?
* How to implement distributed / GPU computations and avoid racing etc.?

# Current tasks
* Implement key backend functions.
* Implement MVP which could fit a model with at least one (`Dense`) layer via `SGD` optimizer.
