
# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import pandas as pd
# import tqdm.notebook as tqdm
import tqdm


# Load the dataset and perform train/test split (matching model_experimentation.py)
# dataset_name = 'four_params_Kbearing_c_finite_width_filtered'
# dataset_name = 'four_params_Kbearing_c'
dataset_name = 'four_params_Kbearing_c_outliers_removed'
data_array = np.load(f'../data/ml_formatted_datasets/{dataset_name}.npy')

print(f"Dataset shape: {data_array.shape}")
print(f"Features: w_over_r, a_over_c, a_over_t, R_over_t")
print(f"Target: Kbearing_c")

# Feature indices (based on your description)
# height_factor_idx = 0
# thickness_idx = 1
w_over_r_idx = 0
a_over_c_idx = 1
a_over_t_idx = 2
R_over_t_idx = 3
Kbearing_c_idx = 4

only_remove_corner_cases = True
if only_remove_corner_cases:
    # Load in the removal_candidate_features.npy file
    removal_candidates = np.load('../data/interim_datasets/removal_candidate_features.npy')
    # Create a mask to filter out removal candidates
    mask = np.ones(len(data_array), dtype=bool)
    for candidate in removal_candidates:
        candidate_mask = np.all(data_array[:, :4] == candidate, axis=1)
        mask = mask & ~candidate_mask
    removal_candidate_data_array = data_array[mask]
    nonremoval_candidate_data_array = data_array[~mask]

    # Shuffle the dataset rows with the same seed as model_experimentation.py
    np.random.seed(42)
    np.random.shuffle(removal_candidate_data_array)
    np.random.seed(42)
    np.random.shuffle(nonremoval_candidate_data_array)

    # Stack the removal candidates on the end of the nonremoval candidates, to create the full dataset array and slice test data off the end
    data_array = np.vstack([nonremoval_candidate_data_array, removal_candidate_data_array])

else:
    # Shuffle the dataset rows with the same seed as model_experimentation.py
    np.random.seed(42)
    np.random.shuffle(data_array)

# Filter out all data points where w_over_r is greater than 50
data_array = data_array[data_array[:, w_over_r_idx] < 10]

# Split the data into 80/20 train/test sets (same as model_experimentation.py)
train_data = data_array[:int(0.8 * len(data_array))]
test_data = data_array[int(0.8 * len(data_array)):]

# Check if any test set data are corner cases
test_corner_cases = []
for candidate in removal_candidates:
    candidate_mask = np.all(test_data[:, :4] == candidate, axis=1)
    if np.any(candidate_mask):
        test_corner_cases.append(test_data[candidate_mask])
print(f"Number of corner cases in test set: {len(test_corner_cases)}")

# Extract features and labels
X_train = train_data[:, :-1]  # First 4 columns: w_over_r, a_over_c, a_over_t, R_over_t
y_train = train_data[:, -1]   # Last column: Kbearing_c
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

print(f"\nTraining data: {X_train.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")

# Display feature ranges
feature_names = ['w_over_r', 'a_over_c', 'a_over_t', 'R_over_t']
print(f"\nFeature ranges in training data:")
for i, name in enumerate(feature_names):
    print(f"{name}: [{X_train[:, i].min():.4f}, {X_train[:, i].max():.4f}]")
print(f"Kbearing_c: [{y_train.min():.4f}, {y_train.max():.4f}]")

# Display the number of unique values for each feature in the training dataset
print(f"\nUnique values in the training dataset:")
for i, name in enumerate(feature_names):
    print(f"{name}: {np.unique(X_train[:, i]).size}")
print(f"Kbearing_c: {np.unique(y_train).size}")

# Plot number of datapoints for each unique w_over_r value
unique_w_over_r, counts_w_over_r = np.unique(X_train[:, w_over_r_idx], return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(unique_w_over_r, counts_w_over_r)
plt.xlabel('w_over_r')
plt.ylabel('Number of datapoints')
plt.title('Distribution of w_over_r in Training Data')
plt.grid()
# plt.show()
print(f"Unique w_over_r: {unique_w_over_r}")
print(f"Counts: {counts_w_over_r}")

# Plot number of datapoints for each unique R_over_t value
unique_R_over_t, counts_R_over_t = np.unique(X_train[:, R_over_t_idx], return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(unique_R_over_t, counts_R_over_t)
plt.xlabel('R_over_t')
plt.ylabel('Number of datapoints')
plt.title('Distribution of R_over_t in Training Data')
plt.grid()
# plt.show()

# Plot number of datapoints for each unique a_over_t value
unique_a_over_t, counts_a_over_t = np.unique(X_train[:, a_over_t_idx], return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(unique_a_over_t, counts_a_over_t)
plt.xlabel('a_over_t')
plt.ylabel('Number of datapoints')
plt.title('Distribution of a_over_t in Training Data')
plt.grid()
# plt.show()

# Plot number of datapoints for each unique a_over_c value
unique_a_over_c, counts_a_over_c = np.unique(X_train[:, a_over_c_idx], return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(unique_a_over_c, counts_a_over_c)
plt.xlabel('a_over_c')
plt.ylabel('Number of datapoints')
plt.title('Distribution of a_over_c in Training Data')
plt.grid()
# plt.show()


# Optimized AFGROW-Style Interpolation Function
def afgrow_interpolation(target_point, training_data, training_labels, feature_names, 
                                   reduction_order=None):
    """
    Optimized AFGROW-style interpolation using vectorized operations.

    Parameters:
    target_point: array of [w_over_r, a_over_c, a_over_t, R_over_t] for prediction
    training_data: X_train array
    training_labels: y_train array
    feature_names: list of feature names
    reduction_order: list of parameter names in order of reduction

    Returns:
    interpolated_value: predicted Kbearing_c
    """

    if reduction_order is None:
        reduction_order = ['w_over_r', 'R_over_t', 'a_over_c', 'a_over_t']

    # Create mapping from parameter names to indices
    param_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Start with full training data
    current_data = training_data.copy()
    current_labels = training_labels.copy()

    # Process each parameter in the reduction order
    for reduction_idx, param_to_reduce in enumerate(reduction_order):
        param_idx = param_to_idx[param_to_reduce]
        target_value = target_point[param_to_idx[param_to_reduce]]

        # Get indices of parameters not yet reduced
        remaining_params = [param_to_idx[p] for p in reduction_order[reduction_idx:]]
        other_params = [p for p in remaining_params if p != param_idx]

        if len(other_params) == 0:
            # Final reduction - just interpolate over the remaining parameter
            if len(current_data) == 1:
                return current_labels[0]

            sort_idx = np.argsort(current_data[:, param_idx])
            sorted_values = current_data[sort_idx, param_idx]
            sorted_labels = current_labels[sort_idx]

            if len(np.unique(sorted_values)) == 1:
                return sorted_labels[0]

            return np.interp(target_value, sorted_values, sorted_labels)

        # Group by unique combinations of other parameters
        # Use a more efficient approach with rounding for floating point comparison
        other_data = np.round(current_data[:, other_params], decimals=10)
        unique_combos, inverse_indices = np.unique(other_data, axis=0, return_inverse=True)

        new_data_list = []
        new_labels_list = []

        # Process each unique combination
        for combo_idx in range(len(unique_combos)):
            # Get all points with this combination
            mask = inverse_indices == combo_idx
            subset_param_values = current_data[mask, param_idx]
            subset_labels = current_labels[mask]

            # Get unique parameter values and their indices
            unique_param_vals, unique_indices = np.unique(subset_param_values, return_index=True)

            if len(unique_param_vals) == 1:
                if np.abs(unique_param_vals[0] - target_value) < 1e-10:
                    interpolated_label = subset_labels[unique_indices[0]]
                else:
                    continue
            else:
                # Sort and interpolate
                sort_order = np.argsort(unique_param_vals)
                sorted_param_vals = unique_param_vals[sort_order]
                sorted_labels = subset_labels[unique_indices[sort_order]]

                interpolated_label = np.interp(target_value, sorted_param_vals, sorted_labels)

            # Build new data point
            new_point = np.zeros(len(feature_names))
            new_point[param_idx] = target_value
            for i, other_idx in enumerate(other_params):
                new_point[other_idx] = unique_combos[combo_idx, i]

            new_data_list.append(new_point)
            new_labels_list.append(interpolated_label)

        if len(new_data_list) > 0:
            current_data = np.array(new_data_list)
            current_labels = np.array(new_labels_list)
        else:
            return np.mean(current_labels)

    return current_labels[0] if len(current_labels) == 1 else np.mean(current_labels)

# Test the AFGROW interpolation function on a few sample test points
print("=== Testing AFGROW Interpolation Function ===\n")

# Select a few test points to demonstrate the interpolation
n_test_samples = 5
sample_indices = np.random.choice(len(X_test), n_test_samples, replace=False)

for i, idx in enumerate(sample_indices):
    test_point = X_test[idx]
    actual_value = y_test[idx]

    print(f"\n{'='*80}")
    print(f"Test Sample #{i+1}")
    print(f"Features: w/r={test_point[0]:.4f}, a/c={test_point[1]:.4f}, a/t={test_point[2]:.4f}, R/t={test_point[3]:.4f}")
    print(f"Actual K value: {actual_value:.6f}")
    print(f"{'='*80}")

    # Test with default reduction order
    pred_default = afgrow_interpolation(test_point, X_train, y_train, feature_names)
    error_default = abs(pred_default - actual_value)
    relative_error_default = (error_default / actual_value) * 100

    print(f"\nDefault order ['a_over_t', 'a_over_c', 'R_over_t', 'w_over_r']:")
    print(f"  Predicted: {pred_default:.6f}")
    print(f"  Error: {error_default:.6f}")
    print(f"  Relative Error: {relative_error_default:.2f}%")

    # Test with alternative reduction order
    alt_order = ['w_over_r', 'R_over_t', 'a_over_c', 'a_over_t']
    pred_alt = afgrow_interpolation(test_point, X_train, y_train, feature_names, reduction_order=alt_order)
    error_alt = abs(pred_alt - actual_value)
    relative_error_alt = (error_alt / actual_value) * 100

    print(f"\nAlternative order {alt_order}:")
    print(f"  Predicted: {pred_alt:.6f}")
    print(f"  Error: {error_alt:.6f}")
    print(f"  Relative Error: {relative_error_alt:.2f}%")

    # Compare the two approaches
    if error_default < error_alt:
        print(f"\n✅ Default order performs better for this point")
    elif error_default > error_alt:
        print(f"\n✅ Alternative order performs better for this point")
    else:
        print(f"\n➡️ Both orders give identical results")

print(f"\n{'='*80}")
print("Testing complete!")
print(f"{'='*80}")


# In[ ]:


# Enhanced Direct 4D Interpolation with multiple interpolation methods
def direct_4d_interpolation(target_point, training_data, training_labels, 
                                     n_neighbors=16, method='multilinear', adaptive=True):
    """
    Perform enhanced 4D interpolation with multiple methods

    Parameters:
    target_point: array of [w_over_r, a_over_c, a_over_t, R_over_t] for prediction
    training_data: X_train array
    training_labels: y_train array
    n_neighbors: number of neighbors for IDW method
    method: 'idw' (inverse distance weighting) or 'multilinear' (true linear interpolation)
    adaptive: if True, adjust n_neighbors based on local density (only for IDW)

    Returns:
    interpolated_value: predicted Kbearing_c
    """

    if method == 'multilinear':
        return _multilinear_interpolation(target_point, training_data, training_labels)
    else:  # method == 'idw'
        return _idw_interpolation(target_point, training_data, training_labels, 
                                 n_neighbors, adaptive)


def _idw_interpolation(target_point, training_data, training_labels, 
                       n_neighbors, adaptive):
    """Inverse distance weighted interpolation"""

    # Normalize features to handle different scales
    feature_ranges = np.ptp(training_data, axis=0)
    feature_ranges[feature_ranges == 0] = 1

    normalized_train = training_data / feature_ranges
    normalized_target = target_point / feature_ranges

    # Calculate Euclidean distances in normalized space
    distances = np.sqrt(np.sum((normalized_train - normalized_target)**2, axis=1))

    # Adaptive neighbor selection
    if adaptive:
        min_distance = np.min(distances)
        if min_distance > 0.5:
            n_neighbors = min(32, len(training_data))
        elif min_distance < 0.1:
            n_neighbors = min(8, len(training_data))

    # Find the n closest neighbors
    n_neighbors = min(n_neighbors, len(training_data))
    closest_indices = np.argsort(distances)[:n_neighbors]
    closest_distances = distances[closest_indices]
    closest_labels = training_labels[closest_indices]

    # Handle exact match
    if closest_distances[0] < 1e-10:
        return closest_labels[0]

    # Inverse distance weighting with power of 2
    weights = 1.0 / (closest_distances**2 + 1e-10)
    weights = weights / np.sum(weights)

    return np.sum(weights * closest_labels)


def _multilinear_interpolation(target_point, training_data, training_labels):
    """
    True multilinear interpolation (2^n vertices in n dimensions)

    For each dimension, finds the nearest neighbors on either side.
    Then performs linear interpolation in all dimensions.
    """

    n_dims = len(target_point)

    # For each dimension, find the bracketing values
    brackets = []  # Will store (lower_val, upper_val) or (exact_val,) for each dimension
    bracket_indices = []  # Indices in training data

    for dim in range(n_dims):
        dim_values = training_data[:, dim]
        unique_vals = np.unique(dim_values)
        target_val = target_point[dim]

        # Find where target falls in the unique values
        idx = np.searchsorted(unique_vals, target_val)

        if idx == 0:
            # Target is below all values - use first value (extrapolation)
            brackets.append((unique_vals[0],))
            bracket_indices.append([(dim, unique_vals[0])])
        elif idx == len(unique_vals):
            # Target is above all values - use last value (extrapolation)
            brackets.append((unique_vals[-1],))
            bracket_indices.append([(dim, unique_vals[-1])])
        elif np.abs(unique_vals[idx-1] - target_val) < 1e-10:
            # Exact match with lower bracket
            brackets.append((unique_vals[idx-1],))
            bracket_indices.append([(dim, unique_vals[idx-1])])
        elif idx < len(unique_vals) and np.abs(unique_vals[idx] - target_val) < 1e-10:
            # Exact match with upper bracket
            brackets.append((unique_vals[idx],))
            bracket_indices.append([(dim, unique_vals[idx])])
        else:
            # Between two values - interpolate
            brackets.append((unique_vals[idx-1], unique_vals[idx]))
            bracket_indices.append([(dim, unique_vals[idx-1]), (dim, unique_vals[idx])])

    # Generate all corner combinations
    # If dimension has 1 value, use it; if 2 values, use both
    corner_combinations = [[]]
    for dim in range(n_dims):
        new_combinations = []
        for combo in corner_combinations:
            for bracket_val in brackets[dim]:
                new_combinations.append(combo + [bracket_val])
        corner_combinations = new_combinations

    # Find actual training points at each corner
    corner_values = []
    corner_coords = []

    for corner in corner_combinations:
        # Find training points that match this corner
        mask = np.ones(len(training_data), dtype=bool)
        for dim in range(n_dims):
            mask &= np.abs(training_data[:, dim] - corner[dim]) < 1e-10

        if np.sum(mask) > 0:
            # Use the mean if multiple points match (shouldn't happen in clean data)
            corner_values.append(np.mean(training_labels[mask]))
            corner_coords.append(corner)

    if len(corner_values) == 0:
        # Fallback: no exact corners found, use nearest neighbor
        distances = np.sqrt(np.sum((training_data - target_point)**2, axis=1))
        return training_labels[np.argmin(distances)]

    if len(corner_values) == 1:
        # Only one corner available
        return corner_values[0]

    # Perform multilinear interpolation
    # This is a recursive linear interpolation across all dimensions
    result = _recursive_linear_interp(target_point, corner_coords, corner_values, 
                                     brackets, 0)

    return result


def _recursive_linear_interp(target_point, corner_coords, corner_values, brackets, dim):
    """
    Recursively perform linear interpolation across dimensions

    Parameters:
    target_point: the point to interpolate at
    corner_coords: list of corner coordinates
    corner_values: list of values at corners
    brackets: bracket information for each dimension
    dim: current dimension being processed
    """

    n_dims = len(target_point)

    if dim == n_dims:
        # Base case: all dimensions processed, return the value
        return corner_values[0] if len(corner_values) == 1 else np.mean(corner_values)

    # Group corners by their value in the current dimension
    if len(brackets[dim]) == 1:
        # Only one value in this dimension, skip interpolation
        return _recursive_linear_interp(target_point, corner_coords, corner_values, 
                                       brackets, dim + 1)

    # Split corners into lower and upper groups
    lower_val, upper_val = brackets[dim]

    lower_corners = []
    lower_values = []
    upper_corners = []
    upper_values = []

    for coord, val in zip(corner_coords, corner_values):
        if np.abs(coord[dim] - lower_val) < 1e-10:
            lower_corners.append(coord)
            lower_values.append(val)
        elif np.abs(coord[dim] - upper_val) < 1e-10:
            upper_corners.append(coord)
            upper_values.append(val)

    # Recursively interpolate on lower and upper subspaces
    if len(lower_values) == 0:
        # No lower bracket, use upper only
        return _recursive_linear_interp(target_point, upper_corners, upper_values, 
                                       brackets, dim + 1)
    if len(upper_values) == 0:
        # No upper bracket, use lower only
        return _recursive_linear_interp(target_point, lower_corners, lower_values, 
                                       brackets, dim + 1)

    # Interpolate in both subspaces
    lower_result = _recursive_linear_interp(target_point, lower_corners, lower_values, 
                                           brackets, dim + 1)
    upper_result = _recursive_linear_interp(target_point, upper_corners, upper_values, 
                                           brackets, dim + 1)

    # Linear interpolation between lower and upper in this dimension
    t = (target_point[dim] - lower_val) / (upper_val - lower_val + 1e-10)
    t = np.clip(t, 0, 1)  # Ensure t is in [0, 1]

    return (1 - t) * lower_result + t * upper_result


# In[15]:


print("=== Interpolation Methods Comparison ===\n")

# Test on all test points
test_points = X_test[:]
actual_values = y_test[:]

afgrow_predictions = []
direct_4d_predictions = []
afgrow_errors = []
direct_4d_errors = []

# Optional: Remove all datapoints with high w/r ratios greater than 15
filter_high_wr = False  # Set to False to include all datapoints
wr_threshold = 15.0

if filter_high_wr:
    # Create mask for datapoints with w/r <= threshold for the test set
    wr_mask = test_points[:, w_over_r_idx] <= wr_threshold
    test_points = test_points[wr_mask]
    actual_values = actual_values[wr_mask]
    print(f"Filtered out {np.sum(~wr_mask)} test datapoints with w/r > {wr_threshold}")
    print(f"Remaining test datapoints: {len(test_points)}")

    # Filter out w/r >= threshold for the training set
    train_wr_mask = X_train[:, w_over_r_idx] <= wr_threshold
    X_train_filtered = X_train[train_wr_mask]
    y_train_filtered = y_train[train_wr_mask]
    print(f"Filtered out {np.sum(~train_wr_mask)} training datapoints with w/r > {wr_threshold}")
    print(f"Remaining training datapoints: {len(X_train_filtered)}")

else:
    print(f"Using all test datapoints: {len(test_points)}")
    X_train_filtered = X_train
    y_train_filtered = y_train
    print(f"Using all training datapoints: {len(X_train_filtered)}")

print(f"Testing on {len(test_points)} test points...")
afgrow_order =  ['w_over_r', 'R_over_t', 'a_over_c', 'a_over_t']

for i, (test_point, actual) in enumerate(tqdm.tqdm(zip(test_points, actual_values), 
                                                     total=len(test_points), 
                                                     desc="Interpolating test points")):
    # N_stop = 100
    # if i == N_stop:
    #     actual_values = actual_values[:N_stop]
    #     break


    # AFGROW interpolation (using filtered training data)
    afgrow_pred = afgrow_interpolation(test_point, X_train_filtered, y_train_filtered, feature_names, reduction_order=afgrow_order
)
    afgrow_error = abs(afgrow_pred - actual)

    # Direct 4D interpolation (using filtered training data)
    direct_4d_pred = direct_4d_interpolation(test_point, X_train_filtered, y_train_filtered, n_neighbors=16)
    direct_4d_error = abs(direct_4d_pred - actual)

    afgrow_predictions.append(afgrow_pred)
    direct_4d_predictions.append(direct_4d_pred)
    afgrow_errors.append(afgrow_error)
    direct_4d_errors.append(direct_4d_error)


# Convert to numpy arrays for easier analysis
afgrow_predictions = np.array(afgrow_predictions)
direct_4d_predictions = np.array(direct_4d_predictions)
afgrow_errors = np.array(afgrow_errors)
direct_4d_errors = np.array(direct_4d_errors)

# Summary statistics for AFGROW
afgrow_mean_error = np.mean(afgrow_errors)
afgrow_max_error = np.max(afgrow_errors)
afgrow_mean_relative_error = np.mean(np.abs((afgrow_predictions - actual_values) / actual_values) * 100)

# Summary statistics for Direct 4D
direct_4d_mean_error = np.mean(direct_4d_errors)
direct_4d_max_error = np.max(direct_4d_errors)
direct_4d_mean_relative_error = np.mean([abs(p - a) / a * 100 for p, a in zip(direct_4d_predictions, actual_values)])

print(f"\n📊 AFGROW Interpolation Performance:")
print(f"Mean Absolute Error: {afgrow_mean_error:.6f}")
print(f"Maximum Error: {afgrow_max_error:.6f}")
print(f"Mean Relative Error: {afgrow_mean_relative_error:.2f}%")

print(f"\n📊 Direct 4D Interpolation Performance:")
print(f"Mean Absolute Error: {direct_4d_mean_error:.6f}")
print(f"Maximum Error: {direct_4d_max_error:.6f}")
print(f"Mean Relative Error: {direct_4d_mean_relative_error:.2f}%")

print(f"\n🔍 Comparison:")
print(f"AFGROW vs Direct 4D - Mean Error Ratio: {afgrow_mean_error/direct_4d_mean_error:.3f}")
print(f"AFGROW vs Direct 4D - Max Error Ratio: {afgrow_max_error/direct_4d_max_error:.3f}")
if afgrow_mean_relative_error < direct_4d_mean_relative_error:
    print(f"✅ AFGROW method performs better (lower relative error)")
else:
    print(f"✅ Direct 4D method performs better (lower relative error)")



# Save the afgrow method predictions
np.savez_compressed("afgrow_interpolation_results.npz",
                        test_points=test_points,
                        actual_values=actual_values,
                        afgrow_predictions=afgrow_predictions,
                        afgrow_errors=afgrow_errors,
                        direct_4d_predictions=direct_4d_predictions,
                        direct_4d_errors=direct_4d_errors)


