import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from orbit import MACHINE_EPS


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print(len(df))

    df = df[df["std_energy_loss"] != 0.0]
    df = df[df["step_size"] != 0.0]
    df["masses"] = df["masses"].apply(ast.literal_eval)
    df["num_objects"] = df["masses"].apply(len)
    df["solver_type"] = df["solver"].str.split("Solver").str[0]
    df["step_size"] = df["step_size"].astype(float)
    df["log_step_size"] = np.log10(df["step_size"])
    df["std_energy_loss"] = df["std_energy_loss"].astype(float)
    df["log_std_energy_loss"] = np.log10(df["std_energy_loss"])

    # df["mae_deviance"] = df["mae_deviance"].astype(float)
    # df["log_mae_deviance"] = np.log10(df["mae_deviance"])

    df = df[df["log_std_energy_loss"] > np.log10(MACHINE_EPS) + 1]

    # df = df[df["log_mae_deviance"] > np.log10(MACHINE_EPS) + 1]

    # df = df[df["lyapunov"] < 0.004]
    print(len(df))
    df.drop(columns="notes", inplace=True, errors="ignore")
    df = remove_outliers(df, "log_std_energy_loss")
    
    # lyapunov details
    
    unique_num_objects = df["num_objects"].unique()

    print(df["lyapunov"].describe())

    for num_objects in unique_num_objects:
        df_subset = df[df["num_objects"] == num_objects]
        print(df_subset["lyapunov"].describe())
    return df


def regression_analysis(df):
    unique_num_objects = df["num_objects"].unique()

    # Create a single plot for all bodies
    plt.figure(figsize=(12, 7))

    for num_objects in unique_num_objects:
        df_subset = df[df["num_objects"] == num_objects]

        for solver in df_subset["solver_type"].unique():
            solver_data = df_subset[df_subset["solver_type"] == solver].copy()

            model = fit_regression(solver_data, "log_step_size", "log_std_energy_loss")
            print_regression_results(model, solver, num_objects)

            # Plot on the combined figure
            plt.scatter(
                solver_data["log_step_size"],
                solver_data["log_std_energy_loss"],
                alpha=0.5,
                label=f"{solver} ({num_objects}-body)",
            )

            x_range = np.linspace(
                solver_data["log_step_size"].min(),
                solver_data["log_step_size"].max(),
                100,
            )
            y_pred = model.params[0] + model.params[1] * x_range
            plt.plot(x_range, y_pred, "--", alpha=0.7)

        plt.xlabel("Log Step Size")
        plt.ylabel("Log Standard Energy Loss")
        plt.title(f"Regression Analysis: {num_objects}-Body")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.close()


def regression_analysis_all(df):
    """Perform regression analysis on all data combined and display a single plot."""
    plt.figure(figsize=(10, 6))

    for solver in df["solver_type"].unique():
        solver_data = df[df["solver_type"] == solver].copy()
        model = fit_regression(solver_data, "log_step_size", "log_std_energy_loss")
        print(f"Overall regression equation for {solver}:")
        print(f"y = {model.params[0]:.4f} + ({model.params[1]:.4f}) * x\n")

        # Plot regression for each solver in the same plot
        x_range = np.linspace(
            solver_data["log_step_size"].min(), solver_data["log_step_size"].max(), 100
        )
        y_pred = model.params[0] + model.params[1] * x_range
        plt.scatter(
            solver_data["log_step_size"],
            solver_data["log_std_energy_loss"],
            alpha=0.5,
            label=f"{solver} (Data)",
        )
        plt.plot(x_range, y_pred, "--", label=f"{solver} (Regression)")

    plt.xlabel("Log Step Size")
    plt.ylabel("Log Standard Energy Loss")
    plt.title("Regression Analysis: All Bodies Combined")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()


def remove_outliers(df, column):
    z_scores = np.abs(stats.zscore(df[column]))
    df = df[z_scores < 3]
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]


def fit_regression(df, x_col, y_col):
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    return sm.OLS(y, X).fit()


def print_regression_results(model, solver, num_objects):
    print(f"Regression equation for {solver} ({num_objects}-body):")
    print(f"y = {model.params[0]:.4f} + ({model.params[1]:.4f}) * x\n")


def plot_regression(df, model, solver):
    x_range = np.linspace(df["log_step_size"].min(), df["log_step_size"].max(), 100)
    y_pred = model.params[0] + model.params[1] * x_range
    plt.scatter(
        df["log_step_size"],
        df["log_std_energy_loss"],
        alpha=0.5,
        label=f"{solver} (Data)",
    )
    plt.plot(x_range, y_pred, "--", label=f"{solver} (Regression)")
    plt.close()


def computational_efficiency_analysis(df):
    for num_objects in df["num_objects"].unique():
        df_subset = df[df["num_objects"] == num_objects]
        plt.figure(figsize=(12, 6))
        for solver in df_subset["solver_type"].unique():
            solver_data = df_subset[df_subset["solver_type"] == solver]
            plt.scatter(
                solver_data["log_step_size"],
                solver_data["execution_duration"],
                label=solver,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Execution Time (s)")
        plt.xlabel("Log Step Size")
        plt.title(f"Execution Time vs Log Step Size for ({num_objects}-Body system)")
        plt.legend()
        plt.show()
        plt.close()


def extract_max_threshold(thresholds):
    thresholds_dict = eval(thresholds)
    return max([k for k, v in thresholds_dict.items() if v is not None], default=0.0)


def compute_solver_scores(df):
    df["max_threshold"] = df["energy_thresholds"].apply(extract_max_threshold)
    scores = (
        df.groupby("solver_type")["execution_duration"].median()
        / df["execution_duration"].median()
        * (1 / 3)
        + df.groupby("solver_type")["std_energy_loss"].median()
        / df["std_energy_loss"].median()
        * (1 / 3)
        + df.groupby("solver_type")["max_threshold"].median()
        / df["max_threshold"].median()
        * (1 / 3)
    ) / df.groupby("solver_type")["std_energy_loss"].count()
    scores = scores.to_dict()
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    print(scores)

    scores = (
        df.groupby("solver")["execution_duration"].median()
        / df["execution_duration"].median()
        * (1 / 3)
        + df.groupby("solver")["std_energy_loss"].median()
        / df["std_energy_loss"].median()
        * (1 / 3)
        + df.groupby("solver")["max_threshold"].median()
        / df["max_threshold"].median()
        * (1 / 3)
    ) / df.groupby("solver")["std_energy_loss"].count()
    scores = scores.to_dict()
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))

def regression_analysis_individual(df, metric):
    # Initialize lists to store average gradients
    avg_gradients = {"solver_type": [], "avg_gradient": []}

    # Iterate over each solver type
    for solver in df["solver_type"].unique():
        solver_data = df[df["solver_type"] == solver]

        # Initialize lists to store gradients for current solver type
        gradients = []

        # Iterate over each experiment ID
        for experiment_id in solver_data["experiment_id"].unique():
            experiment_data = solver_data[solver_data["experiment_id"] == experiment_id]

            # Fit regression model
            model = fit_regression(
                experiment_data, "log_step_size", metric
            )

            # Calculate gradient (coefficient of log_step_size)
            gradient = model.params[1]
            gradients.append(gradient)
        # Calculate average gradient for current solver type
        # avg_gradient = np.mean(gradients)
        avg_gradient = np.median(gradients)
        print(gradients)
        avg_gradients["solver_type"].append(solver)
        avg_gradients["avg_gradient"].append(avg_gradient)

    # Convert to DataFrame
    avg_gradients_df = pd.DataFrame(avg_gradients)

    remove_outliers(avg_gradients_df, "avg_gradient")

    # Plotting average gradients
    plt.figure(figsize=(10, 6))
    print(avg_gradients_df)
    sns.barplot(x="solver_type", y="avg_gradient", data=avg_gradients_df)
    plt.xlabel("Solver Type")
    plt.ylabel(f'Average Gradient of {" ".join([x.capitalize() for x in metric.replace("_", " ").split(" ")])} vs Log Step Size')
    plt.title("Average Gradient of Regression Lines by Solver Type")
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()

    # Plot KDE of gradients per solver with y-axis normalized to max=1
    plt.figure(figsize=(12, 6))

    for solver in df["solver_type"].unique():
        solver_data = df[df["solver_type"] == solver]

        gradients = []
        for experiment_id in solver_data["experiment_id"].unique():
            experiment_data = solver_data[solver_data["experiment_id"] == experiment_id]
            model = fit_regression(
                experiment_data, "log_step_size", "log_std_energy_loss"
            )
            gradients.append(model.params[1])

        # Plot only KDE curve
        kde = sns.kdeplot(gradients, label=solver, fill=False, linewidth=2)

        # Normalize the y-axis to max 1

    plt.xlabel("Gradient")
    plt.ylabel("Density")
    plt.title("KDE of Regression Gradients by Solver Type")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    df = load_and_preprocess_data("experiment_data/experiment.csv")
    regression_analysis_individual(df, "log_std_energy_loss")
    computational_efficiency_analysis(df)
    # compute_solver_scores(df)


if __name__ == "__main__":
    main()
