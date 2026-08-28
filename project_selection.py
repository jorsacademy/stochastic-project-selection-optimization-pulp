import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

Project = Dict[str, float | str]
Projects = Mapping[int, Project]
Dependencies = Mapping[int, Iterable[int]]


def _validate_inputs(
    projects: Projects,
    budget: float,
    dependencies: Optional[Dependencies] = None,
) -> None:
    """Validate project-selection inputs before building the optimization model."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not projects:
        raise ValueError("projects cannot be empty")

    for project_id, project in projects.items():
        missing = {"name", "value", "cost"} - project.keys()
        if missing:
            raise ValueError(
                f"Project {project_id} is missing required fields: {sorted(missing)}"
            )

        cost = float(project["cost"])
        if cost < 0:
            raise ValueError(f"Project {project_id} has a negative cost")

    if dependencies:
        project_ids = set(projects)
        for project_id, required_projects in dependencies.items():
            if project_id not in project_ids:
                raise ValueError(f"Unknown project in dependencies: {project_id}")
            for required_id in required_projects:
                if required_id not in project_ids:
                    raise ValueError(
                        f"Project {project_id} depends on unknown project {required_id}"
                    )


def project_selection(
    projects: Projects,
    budget: float,
    dependencies: Optional[Dependencies] = None,
) -> Tuple[List[int], float, float]:
    """Solve the budget-constrained project-selection problem with PuLP.

    Each project is represented as::

        {project_id: {"name": str, "value": float, "cost": float}}

    If ``dependencies`` contains ``{A: [B, C]}``, selecting project A forces
    projects B and C to be selected as well.
    """
    _validate_inputs(projects, budget, dependencies)

    model = pulp.LpProblem("Project_Selection", pulp.LpMaximize)
    x = {
        project_id: pulp.LpVariable(f"x_{project_id}", cat=pulp.LpBinary)
        for project_id in projects
    }

    model += pulp.lpSum(
        float(projects[project_id]["value"]) * x[project_id]
        for project_id in projects
    )

    model += (
        pulp.lpSum(
            float(projects[project_id]["cost"]) * x[project_id]
            for project_id in projects
        )
        <= budget,
        "Budget_Constraint",
    )

    if dependencies:
        for project_id, required_projects in dependencies.items():
            for required_id in required_projects:
                model += (
                    x[project_id] <= x[required_id],
                    f"Dependency_{project_id}_{required_id}",
                )

    status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        raise RuntimeError(f"Optimization did not reach an optimal solution: {status}")

    selected_projects = [
        project_id
        for project_id in projects
        if pulp.value(x[project_id]) is not None
        and pulp.value(x[project_id]) > 0.5
    ]

    objective_value = float(pulp.value(model.objective) or 0.0)
    used_budget = sum(
        float(projects[project_id]["cost"]) for project_id in selected_projects
    )
    remaining_budget = float(budget - used_budget)

    return selected_projects, objective_value, remaining_budget


def _value_cost_ratio(value: float, cost: float) -> float:
    """Return a safe value/cost ratio, treating zero-cost positive value as infinity."""
    if cost == 0:
        if value > 0:
            return math.inf
        return 0.0
    return value / cost


def visualize_results(
    projects: Projects,
    selected_projects: Iterable[int],
    budget: float,
    used_budget: float,
):
    """Visualize deterministic project-selection results."""
    selected_set = set(selected_projects)

    df = pd.DataFrame(
        [
            {
                "Project": str(projects[project_id]["name"]),
                "Value": float(projects[project_id]["value"]),
                "Cost": float(projects[project_id]["cost"]),
                "Selected": project_id in selected_set,
                "Value/Cost Ratio": _value_cost_ratio(
                    float(projects[project_id]["value"]),
                    float(projects[project_id]["cost"]),
                ),
            }
            for project_id in projects
        ]
    )

    df = df.sort_values(
        ["Selected", "Value/Cost Ratio"], ascending=[False, False]
    ).reset_index(drop=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    selected_value = df.loc[df["Selected"], "Value"].sum()
    not_selected_value = df.loc[~df["Selected"], "Value"].sum()
    ax1.bar(["Selected", "Not Selected"], [selected_value, not_selected_value])
    ax1.set_title("Total Value")
    ax1.set_ylabel("Value")

    ax2.bar(["Used", "Remaining"], [used_budget, budget - used_budget])
    ax2.set_title("Budget Usage")
    ax2.set_ylabel("Budget")

    finite_ratios = df["Value/Cost Ratio"].replace([np.inf, -np.inf], np.nan)
    replacement = finite_ratios.max()
    if pd.isna(replacement):
        replacement = 1.0
    plot_ratios = df["Value/Cost Ratio"].replace(np.inf, replacement * 1.1)

    positions = np.arange(len(df))
    ax3.bar(positions, plot_ratios)
    ax3.set_title("Projects by Value/Cost Ratio")
    ax3.set_ylabel("Value/Cost Ratio")
    ax3.set_xticks(positions)
    ax3.set_xticklabels(df["Project"], rotation=45, ha="right")

    plt.tight_layout()
    return fig


def stochastic_project_selection(
    projects: Projects,
    budget: float,
    dependencies: Optional[Dependencies] = None,
    num_scenarios: int = 1000,
    random_seed: Optional[int] = 42,
) -> dict:
    """Estimate project robustness under uncertain costs and values.

    Costs vary uniformly by +/-20% and values vary uniformly by +/-30%.
    Dependency constraints are preserved in every scenario.
    """
    _validate_inputs(projects, budget, dependencies)
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be a positive integer")

    rng = np.random.default_rng(random_seed)
    selection_frequency = {project_id: 0 for project_id in projects}
    value_distribution: List[float] = []
    budget_usage: List[float] = []

    for _ in range(num_scenarios):
        scenario_projects = {}
        for project_id, project in projects.items():
            cost_factor = rng.uniform(0.8, 1.2)
            value_factor = rng.uniform(0.7, 1.3)
            scenario_projects[project_id] = {
                "name": project["name"],
                "value": float(project["value"]) * value_factor,
                "cost": float(project["cost"]) * cost_factor,
            }

        selected, total_value, remaining = project_selection(
            scenario_projects,
            budget,
            dependencies=dependencies,
        )

        for project_id in selected:
            selection_frequency[project_id] += 1

        value_distribution.append(total_value)
        budget_usage.append(budget - remaining)

    selection_probability = {
        project_id: frequency / num_scenarios
        for project_id, frequency in selection_frequency.items()
    }
    robust_projects = sorted(
        selection_probability.items(), key=lambda item: item[1], reverse=True
    )

    return {
        "selection_probability": selection_probability,
        "robust_projects": robust_projects,
        "value_distribution": value_distribution,
        "budget_usage": budget_usage,
        "mean_value": float(np.mean(value_distribution)),
        "mean_budget_usage": float(np.mean(budget_usage)),
    }


def visualize_stochastic_results(projects: Projects, stochastic_results: dict):
    """Visualize stochastic project-selection results."""
    project_ids = list(projects)
    project_names = [str(projects[project_id]["name"]) for project_id in project_ids]
    probabilities = [
        stochastic_results["selection_probability"][project_id]
        for project_id in project_ids
    ]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    positions = np.arange(len(project_names))
    ax1.bar(positions, probabilities)
    ax1.set_title("Project Selection Probability")
    ax1.set_ylabel("Probability")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(project_names, rotation=45, ha="right")

    ax2.hist(stochastic_results["value_distribution"], bins=20)
    ax2.axvline(
        stochastic_results["mean_value"],
        linestyle="dashed",
        linewidth=2,
        label=f'Mean: {stochastic_results["mean_value"]:.2f}',
    )
    ax2.set_title("Distribution of Total Value")
    ax2.set_xlabel("Total Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    ax3.hist(stochastic_results["budget_usage"], bins=20)
    ax3.axvline(
        stochastic_results["mean_budget_usage"],
        linestyle="dashed",
        linewidth=2,
        label=f'Mean: {stochastic_results["mean_budget_usage"]:.2f}',
    )
    ax3.set_title("Distribution of Budget Usage")
    ax3.set_xlabel("Budget Usage")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    costs = [float(projects[project_id]["cost"]) for project_id in project_ids]
    values = [float(projects[project_id]["value"]) for project_id in project_ids]
    bubble_sizes = [max(probability * 500, 20) for probability in probabilities]

    ax4.scatter(costs, values, s=bubble_sizes, alpha=0.6)
    ax4.set_title("Project Value vs Cost (bubble size = selection probability)")
    ax4.set_xlabel("Cost")
    ax4.set_ylabel("Value")

    for index, project_id in enumerate(project_ids):
        ax4.annotate(str(projects[project_id]["name"]), (costs[index], values[index]))

    plt.tight_layout()
    return fig


def main() -> None:
    projects = {
        1: {"name": "Project A", "value": 100, "cost": 50},
        2: {"name": "Project B", "value": 120, "cost": 80},
        3: {"name": "Project C", "value": 80, "cost": 30},
        4: {"name": "Project D", "value": 150, "cost": 100},
        5: {"name": "Project E", "value": 90, "cost": 40},
        6: {"name": "Project F", "value": 70, "cost": 20},
        7: {"name": "Project G", "value": 200, "cost": 150},
        8: {"name": "Project H", "value": 110, "cost": 60},
    }
    total_budget = 250
    dependencies = {1: [3], 7: [8]}

    selected, total_value, remaining = project_selection(
        projects, total_budget, dependencies
    )
    used_budget = total_budget - remaining

    print(f"Available Budget: ${total_budget}")
    print(f"Selected Projects: {[projects[p]['name'] for p in selected]}")
    print(f"Total Value: {total_value:.2f}")
    print(f"Used Budget: ${used_budget:.2f}")
    print(f"Remaining Budget: ${remaining:.2f}")

    visualize_results(projects, selected, total_budget, used_budget)

    stochastic_results = stochastic_project_selection(
        projects,
        total_budget,
        dependencies=dependencies,
        num_scenarios=1000,
        random_seed=42,
    )

    print("\nStochastic Analysis Results:")
    print(f"Mean Total Value: {stochastic_results['mean_value']:.2f}")
    print(f"Mean Budget Usage: ${stochastic_results['mean_budget_usage']:.2f}")
    print("\nProject Selection Probabilities:")
    for project_id, probability in stochastic_results["robust_projects"]:
        print(f"{projects[project_id]['name']}: {probability * 100:.1f}%")

    visualize_stochastic_results(projects, stochastic_results)
    plt.show()


if __name__ == "__main__":
    main()
