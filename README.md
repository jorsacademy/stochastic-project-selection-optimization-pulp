# Stochastic Project Selection Optimization with PuLP

A compact Python implementation of budget-constrained project portfolio selection using mixed-integer linear programming (MILP), with optional project dependencies and Monte Carlo-style uncertainty analysis.

## What it does

The deterministic model selects the combination of projects that maximizes total value while respecting a fixed budget. Optional dependency constraints can force prerequisite projects to be selected together with dependent projects.

The stochastic analysis repeatedly perturbs project costs and values, solves the optimization model for each scenario, and reports how frequently each project is selected. This provides a simple robustness view under uncertainty.

## Features

- Binary project-selection MILP using PuLP
- Budget constraint
- Project dependency constraints
- Solver-status validation
- Input validation
- Safe handling of zero-cost projects in value/cost ratios
- Reproducible stochastic simulation through `random_seed`
- Selection-probability and distribution visualizations

## Model

For binary decision variable `x_i` indicating whether project `i` is selected:

```text
maximize    sum(value_i * x_i)
subject to  sum(cost_i * x_i) <= budget
            x_i <= x_j   for each dependency i -> j
            x_i in {0, 1}
```

If project `i` depends on project `j`, then selecting `i` requires selecting `j`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Run

```bash
python project_selection.py
```

The bundled example defines eight projects, a budget of 250, and two dependency relationships. It first solves the deterministic problem and then runs 1,000 stochastic scenarios.

## Stochastic assumptions

For each scenario:

- Project costs vary uniformly by +/-20%.
- Project values vary uniformly by +/-30%.
- The original dependency constraints remain active.
- The default random seed is `42` for reproducibility.

You can change these settings directly in `stochastic_project_selection()`.

## Example API usage

```python
from project_selection import project_selection, stochastic_project_selection

projects = {
    1: {"name": "Project A", "value": 100, "cost": 50},
    2: {"name": "Project B", "value": 120, "cost": 80},
    3: {"name": "Project C", "value": 80, "cost": 30},
}

dependencies = {
    1: [3],
}

selected, objective_value, remaining_budget = project_selection(
    projects,
    budget=120,
    dependencies=dependencies,
)

results = stochastic_project_selection(
    projects,
    budget=120,
    dependencies=dependencies,
    num_scenarios=1000,
    random_seed=42,
)
```

## Project structure

```text
.
├── project_selection.py
├── requirements.txt
└── README.md
```

## Notes

PuLP uses the CBC solver by default in this project. Depending on your Python environment and platform, PuLP may install CBC automatically; otherwise, configure an available MILP solver supported by PuLP.
