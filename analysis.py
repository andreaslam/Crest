import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from orbit import MACHINE_EPS


def load_and_preprocess_data(filepath):
    """Loads and preprocesses the experimental data from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Initial record count: {len(df)}")

    # Filter out invalid data points
    df = df[(df["std_energy_loss"] > 0.0) & (df["step_size"] > 0.0)]

    # Type conversion and feature engineering
    df["masses"] = df["masses"].apply(ast.literal_eval)
    df["num_objects"] = df["masses"].apply(len)
    df["solver_type"] = df["solver"].str.split("Solver").str[0]
    df["step_size"] = df["step_size"].astype(float)
    df["std_energy_loss"] = df["std_energy_loss"].astype(float)

    # Log transformations
    df["log_step_size"] = np.log10(df["step_size"])
    df["log_std_energy_loss"] = np.log10(df["std_energy_loss"].clip(lower=MACHINE_EPS))

    # Filter based on machine epsilon
    df = df[df["log_std_energy_loss"] > np.log10(MACHINE_EPS) + 1]

    df.drop(columns="notes", inplace=True, errors="ignore")
    df = remove_outliers(df, "log_std_energy_loss")

    print(f"Final record count after preprocessing: {len(df)}")
    return df


def remove_outliers(df, column):
    """Removes outliers from a DataFrame column using the IQR method."""
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def fit_regression(df, x_col, y_col):
    """Fits an Ordinary Least Squares (OLS) regression model."""
    if df.empty or len(df) < 2:
        return None
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    return sm.OLS(y, X).fit()


def extract_max_threshold(thresholds_str):
    """Extracts the maximum energy threshold from its string representation."""
    try:
        thresholds_dict = ast.literal_eval(thresholds_str)
        valid_thresholds = [k for k, v in thresholds_dict.items() if v is not None]
        return max(valid_thresholds, default=0.0)
    except (ValueError, SyntaxError):
        return 0.0


def _ensure_max_threshold_calculated(df):
    """Ensures the 'max_threshold' column exists, calculating it if needed."""
    if "max_threshold" not in df.columns:
        df["max_threshold"] = df["energy_thresholds"].apply(extract_max_threshold)
    return df


def _format_metric_name(metric):
    """Formats a column name into a human-readable label."""
    return " ".join(word.capitalize() for word in metric.replace("_", " ").split())


def _plot_bar(data, x_col, y_col, title, xlabel, ylabel, **kwargs):
    """Generic helper function to create a bar plot."""
    plt.figure(figsize=(12, 7))
    sns.barplot(x=x_col, y=y_col, data=data)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def _plot_kde(data_dict, title, xlabel):
    """Generic helper function to create a Kernel Density Estimate plot."""
    plt.figure(figsize=(12, 7))
    for label, values in data_dict.items():
        sns.kdeplot(values, label=label, fill=False, linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def regression_analysis_of_gradients(df, metric, group_by_body_count=False):
    """
    Calculates and plots the distribution of regression gradients,
    optionally grouped by the number of bodies.
    """

    def _get_gradients(data):
        gradients_by_solver = {}
        for solver in data["solver_type"].unique():
            solver_data = data[data["solver_type"] == solver]
            gradients = []
            for _, experiment_data in solver_data.groupby("experiment_id"):
                model = fit_regression(experiment_data, "log_step_size", metric)
                if model:
                    gradients.append(model.params[1])
            if gradients:
                gradients_by_solver[solver] = gradients
        return gradients_by_solver

    if group_by_body_count:
        unique_bodies = sorted(df["num_objects"].unique())
        for num_objects in unique_bodies:
            df_subset = df[df["num_objects"] == num_objects]
            gradients_by_solver = _get_gradients(df_subset)
            if not gradients_by_solver:
                continue

            avg_gradients_data = [
                {"solver_type": solver, "avg_gradient": np.median(grads)}
                for solver, grads in gradients_by_solver.items()
            ]
            avg_gradients_df = pd.DataFrame(avg_gradients_data)
            avg_gradients_df = remove_outliers(avg_gradients_df, "avg_gradient")

            title_prefix = f"{num_objects}-Body"
            _plot_bar(
                data=avg_gradients_df,
                x_col="solver_type",
                y_col="avg_gradient",
                title=f"{title_prefix}: Average Regression Gradient by Solver",
                xlabel="Solver Type",
                ylabel=f"Median Gradient of {_format_metric_name(metric)}",
            )
    else:
        gradients_by_solver = _get_gradients(df)
        avg_gradients_data = [
            {"solver_type": solver, "avg_gradient": np.median(grads)}
            for solver, grads in gradients_by_solver.items()
        ]
        avg_gradients_df = pd.DataFrame(avg_gradients_data)
        avg_gradients_df = remove_outliers(avg_gradients_df, "avg_gradient")

        _plot_bar(
            data=avg_gradients_df,
            x_col="solver_type",
            y_col="avg_gradient",
            title="Average Gradient of Regression Lines by Solver Type",
            xlabel="Solver Type",
            ylabel=f"Median Gradient of {_format_metric_name(metric)}",
        )

        _plot_kde(
            gradients_by_solver,
            "KDE of Regression Gradients by Solver Type",
            "Gradient",
        )


def plot_energy_threshold_bars(df, group_by="num_objects"):
    """Plots the count of max energy thresholds, grouped by a given column."""
    df = _ensure_max_threshold_calculated(df)
    gb_title = _format_metric_name(group_by)
    unique_groups = sorted(df[group_by].unique())

    fig, axes = plt.subplots(
        len(unique_groups),
        1,
        figsize=(10, 2.5 * len(unique_groups)),
        sharey=True,
        sharex=True,
    )
    if len(unique_groups) == 1:
        axes = [axes]

    fig.suptitle(f"Counts of Max Energy Thresholds by {gb_title}", fontsize=16)

    for ax, grp in zip(axes, unique_groups):
        subset = df[df[group_by] == grp]
        counts = subset["max_threshold"].value_counts().sort_index()
        labels = [f"{th:.0e}" for th in counts.index]
        ax.bar(labels, counts.values, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title(f"{gb_title}: {grp}")
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Max Energy Threshold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def computational_efficiency_analysis(df):
    """Plots execution duration vs. log step size, faceted by number of objects and solver."""
    for num_objects in df["num_objects"].unique():
        df_subset = df[df["num_objects"] == num_objects]
        solvers = df_subset["solver_type"].unique()
        n = len(solvers)
        if n == 0:
            continue

        fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        fig.suptitle(
            f"Computational Efficiency for {num_objects}-Body Systems", fontsize=16
        )
        for ax, solver in zip(axes, solvers):
            data = df_subset[df_subset["solver_type"] == solver]
            ax.scatter(
                data["log_step_size"], data["execution_duration"], marker="x", alpha=0.5
            )
            ax.set_yscale("log")
            ax.set_ylabel("Execution Time (s)")
            ax.set_title(f"{solver}")

        axes[-1].set_xlabel("Log Step Size")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def quadratic_regression_and_pmcc(df):
    """
    Fits a quadratic model to execution duration vs. number of bodies for each solver
    and calculates the Pearson correlation coefficient.
    """
    plt.figure(figsize=(12, 7))
    results = []

    for solver in df["solver_type"].unique():
        df_solver = df[df["solver_type"] == solver]
        grouped = (
            df_solver.groupby("num_objects")["execution_duration"]
            .median()
            .reset_index()
        )

        if len(grouped) < 3:
            continue  # Need at least 3 points for a quadratic fit

        x = grouped["num_objects"].values
        y = grouped["execution_duration"].values

        # Fit quadratic: y = ax^2 + bx + c
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)

        # Pearson correlation
        r, p = stats.pearsonr(x, y)
        results.append(
            {"solver": solver, "coeffs": coeffs, "pearson_r": r, "p_value": p}
        )

        # Plotting
        x_fit = np.linspace(min(x), max(x), 100)
        plt.plot(x_fit, poly(x_fit), label=f"{solver} (r={r:.2f})")
        plt.scatter(x, y, s=50, alpha=0.7)

    plt.xlabel("Number of Bodies")
    plt.ylabel("Median Execution Duration (s)")
    plt.title("Quadratic Fit: Execution Time vs. Number of Bodies", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    for res in results:
        a, b, c = res["coeffs"]
        print(f"{res['solver']}:")
        print(f"  Quadratic Fit: y = {a:.4f}n² + {b:.4f}n + {c:.4f}")
        print(f"  Pearson r = {res['pearson_r']:.4f}, p = {res['p_value']:.4g}\n")


def compute_and_print_solver_scores(df):
    """Calculates and prints composite scores for solvers based on performance metrics."""
    df = _ensure_max_threshold_calculated(df)

    def _calculate_scores(data, group_by_col):
        if data.empty:
            return {}

        grouped = data.groupby(group_by_col)
        median_duration = grouped["execution_duration"].median()
        median_loss = grouped["std_energy_loss"].median()
        median_threshold = grouped["max_threshold"].median()
        count = grouped["std_energy_loss"].count()

        score = (
            (
                (median_duration / data["execution_duration"].median())
                + (median_loss / data["std_energy_loss"].median())
                + (median_threshold / data["max_threshold"].median())
            )
            / 3
            / count
        )

        return dict(sorted(score.to_dict().items(), key=lambda item: item[1]))

    for group_col in ["solver_type", "solver"]:
        scores = _calculate_scores(df, group_col)
        print(f"Scores grouped by {group_col}:")
        print(scores)
        print("-" * 30)


def main():
    """Main function to run the complete analysis pipeline."""
    filepath = "experiment_data/experiment.csv"
    df = load_and_preprocess_data(filepath)

    # --- Regression Analysis ---
    regression_analysis_of_gradients(
        df, "log_std_energy_loss", group_by_body_count=False
    )
    regression_analysis_of_gradients(
        df, "log_std_energy_loss", group_by_body_count=True
    )

    # --- Energy Threshold Analysis ---
    plot_energy_threshold_bars(df, group_by="num_objects")
    plot_energy_threshold_bars(df, group_by="solver_type")

    # --- Execution Time & Efficiency Analysis ---
    computational_efficiency_analysis(df)
    quadratic_regression_and_pmcc(df)

    # --- Solver Scoring ---
    compute_and_print_solver_scores(df)


if __name__ == "__main__":
    main()
