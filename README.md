# Reproducing the Brain Tumor Resection Experiments


This repository contains patented technology. Access is restricted to peer-reviewers for evaluation purposes only. Unauthorized reproduction, modification, or distribution is strictly prohibited.

This repository contains the code, trained graph neural network (GNN)
checkpoints, saved problem instances, archived reference solutions,
configuration files, and reporting utilities needed to reproduce the
computational experiments reported in:

**Optimization Over Trained Graph Neural Networks with an Application in
Brain Tumor Resection**

The repository is designed to support both lightweight verification of
the reported results and full reruns of the optimization experiments.

A reviewer can use the repository to:

-   verify the integrity of the supplied experimental inputs and
    archived solutions;
-   reproduce the fixed-propagation stability analysis;
-   independently recompute graph-functionality values from archived
    resection plans;
-   regenerate aggregate numerical summaries and figures;
-   rerun the single-objective GNN-based and exact optimization models;
-   rerun the bi-objective Pareto-frontier experiments;
-   rerun the real-patient and HCP case-study experiments;
-   reproduce the geometric-expansion benchmark;
-   reconstruct the cross-connectome evaluation.

The repository contains the exact saved problem instances used in the
reported experiments and six trained GNN checkpoints.

It does **not** contain the original GNN training datasets, train/test
splits, or GNN training routines. The supplied checkpoints are therefore
used for inference and optimization, but the original model-training
process itself cannot be rerun from this repository.

No command in this repository generates new tumor instances. All
reproduction experiments use the fixed saved instances supplied in
`instances/`.

## 1. Environment Setup

The reproduction package has been tested with:

-   Python 3.13.5
-   macOS ARM64
-   PyTorch Geometric 2.5.3

The experiments reported in the paper were originally run using Python
3.10.11 and PyTorch Geometric 2.5.3. A complete lock file for the
historical environment is not available. The supplied `requirements.txt`
records the dependency versions tested with this reproduction package.

From the repository directory, create a virtual environment and install
the dependencies:

``` sh
cd "Paper Reproduction"
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If the repository is stored under a different directory name, simply
replace `"Paper Reproduction"` with the corresponding path.

The reproduction scripts resolve repository paths internally, so the
commands can also be executed from other working directories once the
environment has been created.

## 2. Recommended Reviewer Workflow

A reviewer who wants to verify the results without rerunning
computationally expensive optimization models can begin with the
following three commands:

``` sh
.venv/bin/python reproduce.py verify
.venv/bin/python reproduce.py stability
.venv/bin/python reproduce.py summarize
```

These commands do **not** rerun the optimization models.

They perform the following tasks:

1.  `verify` checks the integrity of the supplied inputs and archived
    single-objective solutions;
2.  `stability` reproduces the fixed-propagation stability experiment;
3.  `summarize` independently recalculates numerical summaries and
    regenerates plots from archived resection plans.

The `summarize` command reads archived solver times. These are the
timings recorded during the original optimization experiments and are
**not** timings of a newly executed optimization run.

This three-command workflow is therefore the recommended starting point
for reviewing the numerical results.

## 3. Full Optimization Reproduction

Rerunning the optimization models requires a working Gurobi installation
and a license capable of solving the supplied models.

A size-limited Gurobi license is insufficient for some of the
formulations in this repository.

The GNN inference used in the optimization models runs on CPU.

Full experiment suites can take several hours or longer depending on
hardware, operating system, Gurobi version, and solver behavior.

## 4. Verify the Supplied Inputs and Archived Results

Run:

``` sh
.venv/bin/python reproduce.py verify
```

This command performs consistency and integrity checks on the supplied
experimental inputs and archived single-objective solutions.

The package contains provenance information and expected SHA-256 hashes
for copied inputs. These records are stored in:

``` text
config/provenance.json
```

The verification step is useful for confirming that the supplied
instances and other fixed inputs have not been modified.

## 5. Reproduce the Fixed-Propagation Stability Experiment

Run:

``` sh
.venv/bin/python reproduce.py stability
```

The implementation is located in:

``` text
src/stability.py
```

The experiment uses archived resection plans stored under:

``` text
reference/EpsilonResultsFixed/
```

The exact observations retained for the stability analysis are specified
in:

``` text
config/stability_observations.json
```

The supplied numerical reference workbook is:

``` text
reference/workbooks/constant_adjacency_experiment.xlsx
```

The stability analysis contains **1,052 retained metric observations**.

These observations include repeated resection plans and evaluations
associated with different objectives. They should **not** be
deduplicated.

This command reproduces the fixed-propagation analysis used in the
stability results reported in the paper.

## 6. Rebuild Numerical Summaries and Figures from Archived Solutions

Run:

``` sh
.venv/bin/python reproduce.py summarize
```

This command reconstructs reported summaries from archived resection
plans.

For the single-objective experiments, the script does not rely only on
the objective values saved in the result files. It independently
recalculates the three true global-efficiency objectives from the
archived selected-node sets.

The archived single-objective comparison data are stored in:

``` text
reference/surrogate_vs_original.json
```

This file contains:

``` text
35 instances × 3 global-efficiency objectives = 105 comparisons
```

The command also regenerates the numerical summaries and plots supported
by the archived results.

The solver times reported by `summarize` are archived historical times.
They should not be interpreted as timings of a new solve.

## 7. Single-Objective Optimization Experiments

The single-objective experiments compare the GNN-based optimization
formulation with the exact baseline for three objectives:

-   whole-brain global efficiency;
-   visual-subnetwork global efficiency;
-   MDEN-subnetwork global efficiency.

The saved single-objective instances have IDs 174 through 208.

### 7.1 Run One Saved Instance

For example, to solve Instance 174 for all three global-efficiency
objectives using both the GNN-based formulation and the exact baseline:

``` sh
.venv/bin/python reproduce.py single --instance 174 --method both
```

### 7.2 Run All Single-Objective Instances

To run all 35 saved single-objective instances:

``` sh
.venv/bin/python reproduce.py single --method both
```

This executes all three global-efficiency objectives for each saved
instance.

### 7.3 Main Implementation

The main experiment implementation is located in:

``` text
src/experiments.py
```

The archived comparison results are stored in:

``` text
reference/surrogate_vs_original.json
```

## 8. Bi-Objective Pareto-Frontier Experiments

The bi-objective experiments construct Pareto frontiers involving
whole-brain global efficiency and a second brain-functionality
objective.

The selected experiment configurations used in the paper are specified
in:

``` text
config/biobjective.json
```

The mapping between paper instance numbers and stored instance IDs is
contained in:

``` text
config/instances.json
```

### 8.1 Representative Paper Instance

Paper Instance 20 is the representative visual-network instance used for
the Pareto-frontier illustration.

Run:

``` sh
.venv/bin/python reproduce.py biobjective --paper-id 20 --method both
```

Paper Instance 20 corresponds to:

``` text
instances/mip_instance_118.pkl
```

### 8.2 Run All Reported Bi-Objective Configurations

To rerun all 27 selected bi-objective configurations used in the paper:

``` sh
.venv/bin/python reproduce.py biobjective --method both
```

Archived Pareto-frontier results are stored under:

``` text
reference/EpsilonResultsFixed/
```

If supplied, the archived anatomical visualization panels are stored
under:

``` text
reference/anatomy/
```

These anatomical panels visualize archived solutions. Their presence
should not be interpreted as evidence that a new optimization run has
been performed.

## 9. Real-Patient and HCP Case Study

The case study uses two saved connectomes:

-   stored Instance 168: real patient;
-   stored Instance 170: HCP subject 134829.

Both instances preserve the same tumor geometry, allowing the effect of
the underlying connectome on optimized resection plans to be evaluated.

### 9.1 Rerun the Optimization-Based Case Study

Run:

``` sh
.venv/bin/python reproduce.py case-study --method both
```

This reruns the case-study optimization experiments at the saved
resection-volume thresholds using both supported optimization methods
where applicable.

### 9.2 Reproduce the Geometric-Expansion Benchmark and Related Analyses

Run:

``` sh
.venv/bin/python reproduce.py geometry
```

This command reconstructs:

-   the geometric-expansion benchmark;
-   case-study Pareto coordinate tables;
-   the cross-connectome evaluation.

Archived case-study results are stored under:

``` text
reference/SurgeryEpsilonResults/
```

## 10. Cross-Connectome Evaluation

The cross-connectome analysis evaluates resection plans optimized on one
connectome on both:

-   the real-patient connectome;
-   the HCP comparator connectome.

The corresponding values are reconstructed programmatically from the
saved GE--MDEN Pareto-frontier results.

The reconstruction logic is implemented in:

``` text
src/reporting.py
```

The plan-selection configuration is documented in:

``` text
config/table21.json
```

The relevant saved resection-volume threshold is:

``` text
69.69441243166938%
```

The cross-connectome analysis can be reproduced with:

``` sh
.venv/bin/python reproduce.py geometry
```

## 11. Solver Time Limits

Optimization commands support the option:

``` sh
--time-limit
```

This specifies the per-solve time limit in seconds.

The default value is:

``` text
1800 seconds
```

For example:

``` sh
.venv/bin/python reproduce.py single \
    --instance 174 \
    --method both \
    --time-limit 1800
```

Pareto-frontier generation additionally uses a maximum
frontier-generation limit of:

``` text
7200 seconds
```

Incomplete runs raise an error and are not silently reported as optimal.

Model construction may take several minutes before the solver time limit
begins. The full experiment suites may therefore require many hours.

## 12. Output Files

Newly generated optimization results are written to:

``` text
outputs/reruns/
```

Archived reference results are never overwritten.

Repeating the same reproduction command replaces the corresponding rerun
output file for that command.

Other generated material under `outputs/` includes:

-   verification results;
-   recomputed numerical summaries;
-   generated figures;
-   newly rerun experiment results.

## 13. Paper-to-Code Guide

The following mapping indicates which reproduction commands correspond
to the main computational results in the paper.

### Fixed-Propagation Stability Analysis

Relevant manuscript material:

-   Section 4.2
-   Figure 2
-   Table 4
-   Appendix H

Run:

``` sh
.venv/bin/python reproduce.py stability
```

Main implementation:

``` text
src/stability.py
```

Configuration:

``` text
config/stability_observations.json
```

### Single-Objective Exact-Baseline Comparison

Relevant manuscript material:

-   Section 4.3
-   Figures 3, 9, and 10
-   Tables 5--9

One example instance:

``` sh
.venv/bin/python reproduce.py single --instance 174 --method both
```

Full experiment set:

``` sh
.venv/bin/python reproduce.py single --method both
```

Main implementation:

``` text
src/experiments.py
```

Archived results:

``` text
reference/surrogate_vs_original.json
```

### Bi-Objective Pareto-Frontier Experiments

Relevant manuscript material:

-   Section 4.4
-   Figures 4--6
-   Tables 10--11

Representative experiment:

``` sh
.venv/bin/python reproduce.py biobjective \
    --paper-id 20 \
    --method both
```

Full reported experiment set:

``` sh
.venv/bin/python reproduce.py biobjective --method both
```

Configuration:

``` text
config/biobjective.json
```

Paper-instance mapping:

``` text
config/instances.json
```

Archived results:

``` text
reference/EpsilonResultsFixed/
```

### Real-Patient and HCP Case Study

Relevant manuscript material:

-   Section 5
-   Figure 8
-   Tables 12--20

Optimization experiments:

``` sh
.venv/bin/python reproduce.py case-study --method both
```

Geometric expansion and cross-connectome analysis:

``` sh
.venv/bin/python reproduce.py geometry
```

Archived results:

``` text
reference/SurgeryEpsilonResults/
```

### Cross-Connectome Evaluation

The cross-connectome results are reconstructed using:

``` text
src/reporting.py
```

with:

``` text
config/table21.json
```

Run:

``` sh
.venv/bin/python reproduce.py geometry
```

## 14. Repository Structure

### `instances/`

Contains 57 saved problem-instance files.

Depending on the instance, these contain:

-   structural brain graphs;
-   spatial-adjacency graphs;
-   expected tumor volumes;
-   tumor metadata;
-   ROI ordering information.

These are fixed experimental inputs.

No reproduction command generates new tumor instances.

### `weights/`

Contains six trained GNN checkpoints:

-   four checkpoints used in the optimization experiments;
-   two distinct historical checkpoints used in the stability
    experiment.

Checkpoint assignment is explicit in:

``` text
src/models.py
```

The code does not select checkpoints according to file modification time
or automatically choose the newest checkpoint.

### `src/`

Contains the main reproduction implementation, including:

-   GNN inference;
-   GNN-to-MILP embedding;
-   exact global-efficiency optimization;
-   experiment drivers;
-   stability analysis;
-   validation utilities;
-   result summarization;
-   reporting utilities.

The file:

``` text
generateGraphs.py
```

is retained because some saved Python objects depend on its historical
class definitions and parcel metadata when unpickled.

It is **not** used to generate new experimental tumor instances or to
train GNNs.

### `config/`

Contains experiment definitions and metadata, including:

-   experiment selections;
-   paper-instance mappings;
-   stability observation selections;
-   ROI parcel membership;
-   table-specific configuration;
-   provenance and hash information.

### `reference/`

Contains archived experiment outputs, including:

-   selected-node sets;
-   objective values;
-   solver information;
-   Pareto-frontier results;
-   JSON files;
-   reference workbooks.

Some historical JSON files contain non-standard `NaN` values. Python can
read these archived files.

Newly generated JSON output uses standard JSON `null` values instead.

### `outputs/`

Contains generated reproduction outputs, including:

-   verification results;
-   recomputed summaries;
-   regenerated figures;
-   new optimization reruns.

New optimization results are written under:

``` text
outputs/reruns/
```

## 15. Numerical Conventions

Several implementation conventions are important when comparing newly
generated values with the archived results.

### 15.1 Global Efficiency

The historical global-efficiency implementation retains all 360 node
positions after resection.

Resected nodes are isolated rather than removed from the 360-position
representation.

For subnetwork global efficiency:

-   source and destination endpoint pairs are restricted to the relevant
    subnetwork;
-   shortest paths may traverse the full structural graph;
-   the denominator is restricted to the relevant subnetwork;
-   paths longer than five edges contribute zero.

This is the convention used by the exact baseline and by the archived
reported results.

### 15.2 GNN Architecture

The GNN uses:

1.  one graph-convolution layer with 360 input features and 10 output
    features;
2.  ReLU activation;
3.  global mean pooling;
4.  a scalar linear readout.

Subnetwork GNNs retain all 360 feature columns.

For subnetwork GNN prediction, propagation is performed on the induced
subnetwork graph.

### 15.3 Fixed-Propagation Graph

The default GNN-based optimization formulation uses a fixed propagation
graph.

The graph is constructed by:

1.  removing the edges associated with all candidate tumorous nodes;
2.  restoring self-loops.

This is the fixed-propagation approximation studied in the stability
analysis.

### 15.4 Modularity

For modularity prediction, the GNN input additionally includes the
initial community co-membership matrix.

Post-resection modularity is evaluated relative to the fixed initial
community partition.

### 15.5 ROI Indexing

For optimization node `j`:

``` text
new_order[j]
```

is the zero-based original ROI index represented by that optimization
node.

The corresponding anatomical ROI ID is:

``` text
new_order[j] + 1
```

Selected nodes saved in archived optimization results are expressed
using optimization indices, not anatomical ROI IDs.

### 15.6 Geometric Expansion

The geometric-expansion benchmark:

-   always includes every tumor-core ROI;
-   adds peripheral ROIs according to their stored distance to the
    nearest core ROI.

The stored distance convention is used exactly as supplied by the saved
instances.

## 16. Solver Settings

The GNN-based optimization models retain the historical solver settings:

``` text
MIPGap       = 0
IntFeasTol   = 1e-9
Aggregate    = 0
NumericFocus = 3
```

The exact optimization model likewise preserves its historical solver
configuration.

Solver version, processor architecture, operating system, and available
hardware can affect:

-   model-construction time;
-   solver time;
-   branch-and-bound behavior;
-   the particular optimal solution returned when multiple optimal
    solutions exist.

For this reason, reproduction should primarily compare:

-   objective values;
-   feasibility;
-   optimization status;
-   Pareto-frontier quality;
-   selected-node overlap where appropriate.

Exact agreement in wall-clock time should not be expected across
machines.

Likewise, identical selected-node sets should not necessarily be
expected if multiple optimal solutions exist.

## 17. Results That Cannot Be Reproduced from This Repository

The original GNN training datasets and train/test split are not
included.

Consequently, the following historical results cannot be regenerated
from the supplied checkpoints alone:

-   GNN training-instance counts;
-   the GNN training procedure;
-   prediction MAPEs calculated from the original training/test
    datasets;
-   the associated historical training results reported in Section 4.1
    and Table 3.

The supplied trained checkpoints are sufficient for the inference,
stability, and optimization experiments contained in this reproduction
package.

## 18. Known Differences and Reproducibility Notes

Before interpreting small differences between newly reconstructed
values, archived files, and manuscript tables, please read:

``` text
REPRODUCIBILITY_NOTES.md
```

This file documents known numerical and historical discrepancies
relevant to reproduction.

The repository intentionally preserves the archived experimental
evidence.

Archived data are **not** modified to force agreement with manuscript
tables.

## 19. Compact Command Reference

### Verify inputs and archived solutions

``` sh
.venv/bin/python reproduce.py verify
```

### Reproduce the stability analysis

``` sh
.venv/bin/python reproduce.py stability
```

### Rebuild summaries and figures from archived plans

``` sh
.venv/bin/python reproduce.py summarize
```

### Rerun one single-objective instance

``` sh
.venv/bin/python reproduce.py single \
    --instance 174 \
    --method both
```

### Rerun all single-objective experiments

``` sh
.venv/bin/python reproduce.py single --method both
```

### Rerun the representative bi-objective experiment

``` sh
.venv/bin/python reproduce.py biobjective \
    --paper-id 20 \
    --method both
```

### Rerun all reported bi-objective experiments

``` sh
.venv/bin/python reproduce.py biobjective --method both
```

### Rerun the real-patient and HCP case-study optimization experiments

``` sh
.venv/bin/python reproduce.py case-study --method both
```

### Reproduce geometric expansion and cross-connectome analyses

``` sh
.venv/bin/python reproduce.py geometry
```

### Set a custom per-solve time limit

``` sh
.venv/bin/python reproduce.py single \
    --instance 174 \
    --method both \
    --time-limit 1800
```

## 20. Suggested Reproduction Levels

Depending on the desired level of verification, the repository supports
three useful reproduction levels.

### Level 1: Integrity and Numerical Verification

No optimization reruns are required.

``` sh
.venv/bin/python reproduce.py verify
.venv/bin/python reproduce.py stability
.venv/bin/python reproduce.py summarize
```

This is the fastest way to verify the supplied data, recompute
functionality metrics, reproduce the stability analysis, and regenerate
aggregate reported results.

### Level 2: Targeted Optimization Reproduction

A reviewer can rerun representative experiments without executing the
complete experiment suite.

For example:

``` sh
.venv/bin/python reproduce.py single \
    --instance 174 \
    --method both
```

and:

``` sh
.venv/bin/python reproduce.py biobjective \
    --paper-id 20 \
    --method both
```

This provides a direct check of both the single-objective and
bi-objective optimization pipelines.

### Level 3: Full Optimization Reproduction

To rerun the complete computational experiment set:

``` sh
.venv/bin/python reproduce.py single --method both

.venv/bin/python reproduce.py biobjective --method both

.venv/bin/python reproduce.py case-study --method both

.venv/bin/python reproduce.py geometry
```

Full reproduction may require many hours depending on hardware and
solver configuration.
