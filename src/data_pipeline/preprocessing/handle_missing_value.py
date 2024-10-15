import numpy as np


def get_variance_of_metrics(metrics: np.array) -> np.array:
    cls_nan = dict()
    variance_metric = np.empty((metrics.shape[0], metrics.shape[2], 3), dtype=np.float32)
    for i, element in enumerate(metrics):
        sub_variance_metric = np.empty((element.shape[1], 3), dtype=np.float32)

        for j in range(element.shape[1]):
            variance = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

            # If there are more than 1 non-NaN values, calculate variance
            if np.count_nonzero(~np.isnan(element[:, j])) > 1:
                variance = np.nanvar(element[:, j], axis=0)
                print(variance)
            else:
                if i not in cls_nan:
                    cls_nan[i] = []
                cls_nan[i].append(j)
            # convert to list
            sub_variance_metric[j] = variance
        # print(sub_variance_metric)
        variance_metric[i] = sub_variance_metric

    return variance_metric, cls_nan


# Function to fill NaN values based on the described cases
def handle_fill_nan_by_variance(element, variance_metric):
    old_value = None
    k = 0

    for i in range(element.shape[0]):
        if np.isnan(element[i]).all():
            # Case 1: If element[i] doesn't have a value before, then get the value from the nearest posterior
            # element, and adjust by adding or subtracting the variance of the metric.
            if old_value is None:
                for l in range(i + 1, element.shape[0]):
                    if not np.isnan(element[l]).all():
                        element[i] = element[l] + abs(i - l) * variance_metric
                        break
            else:
                # Case 2: If element[i] doesn't have a value after, then get the value from the nearest previous
                # element, and adjust by adding or subtracting the variance of the metric.
                flat = False
                for l in range(i + 1, element.shape[0]):
                    if not np.isnan(element[l]).all():
                        if l != k:
                            element[i] = old_value + (element[l] - old_value) / ((i - k) / (l - k))
                            flat = True
                            break
                if not flat:
                    # Case 3: If element[i] has values both before and after, then use the nearest previous and
                    # posterior elements
                    element[i] = old_value - abs(i - k) * variance_metric
        old_value = element[i]
    return element
