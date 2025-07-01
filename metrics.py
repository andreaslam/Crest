import numpy as np


def calculate_trajectory_deviance_pair(trajectories, metric):
    """
    Calculate the deviance between two sets of trajectories using the specified metric.
    Supported metrics:
      - 'mae': Mean Absolute Error (average Euclidean distance)
      - 'mse': Mean Squared Error (average of squared Euclidean distances)
      - 'rmse': Root Mean Squared Error (square root of the average squared Euclidean distance)

    Parameters:
      trajectories (tuple): Two dictionaries mapping object IDs to lists of positions.
      metric (str): The metric to use ('mae', 'mse', or 'rmse').

    Returns:
      tuple: (aggregated_error, list_of_errors_per_time_step)
    """
    supported_metrics = {"mae", "mse", "rmse"}
    if metric not in supported_metrics:
        raise ValueError(f"{metric} not supported! Only supports {supported_metrics}")

    t1, t2 = trajectories
    # print("t1", t1)
    # print("t2", t2)
    all_errors = []
    # Loop over each object; assumes both trajectories have the same keys.
    for obj_id in t1:
        traj1 = np.array(t1[obj_id])
        traj2 = np.array(t2[obj_id])
        # Compute Euclidean distance error at each time step along the spatial dimensions.
        errors = np.linalg.norm(traj1 - traj2, axis=1)
        all_errors.append(errors)

    # Concatenate errors from all objects into a single array.
    all_errors = np.concatenate(all_errors)

    if metric == "mae":
        aggregated_error = np.mean(all_errors)
    elif metric == "mse":
        aggregated_error = np.mean(all_errors**2)
    elif metric == "rmse":
        aggregated_error = np.sqrt(np.mean(all_errors**2))

    return aggregated_error, all_errors.tolist()


def load_orbit_data(path):
    """
    Load orbit data from a CSV file with the expected format.
    The CSV file should have a header row and columns where the first column represents the object ID,
    and subsequent columns represent position coordinates.

    Parameters:
      path (str): The base path to the file (without the '_positions.csv' suffix).

    Returns:
      dict: A dictionary mapping each unique object ID to an array of position coordinates.
    """
    data = np.loadtxt(f"{path}_positions.csv", delimiter=",", skiprows=1, dtype=str)
    unique_ids = np.unique(data[:, 0])  # first column: mass id
    return {
        uid: np.array(data[data[:, 0] == uid, 1:].astype(float)) for uid in unique_ids
    }
