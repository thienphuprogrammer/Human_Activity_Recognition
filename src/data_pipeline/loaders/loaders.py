from src.data_pipeline.preprocessing.video_pose_extractor import *
from src.data_pipeline.preprocessing.handle_dim_sequence import *
from src.data_pipeline.preprocessing.handle_missing_value import *


def load_har_dataset(dataset_path: str, resize_dataset_video_path: str,
                     dataset_csv_path: str, max_dim=35):
    # Resize videos and save them
    resize_videos_and_save(dataset_path, resize_dataset_video_path)
    # Process videos and save data to csv
    X, y = process_videos(resize_dataset_video_path, dataset_csv_path)

    X_train_temp, y_train_temp = [], []

    # Pad all sequences to the same length
    for i, element in enumerate(X):
        print(f"Before padding, element {i} shape: {np.shape(element)}")
        padded_element = pad_and_truncate(element, max_dim)
        print(f"After padding, element {i} shape: {np.shape(padded_element)}")
        X_train_temp = X_train_temp + padded_element
        y_train_temp = y_train_temp + [y[i]] * len(padded_element)
        print(f"After padding X_train_temp, element {i} shape: {np.shape(X_train_temp)}")
        print(f"After padding Y_train_temp, element {i} shape: {np.shape(y_train_temp)}")
        print('-----------------------------------')

    # Convert to numpy
    X_train_temp = np.array(X_train_temp)

    # Create empty 4D array with 3 channels
    X_train = np.full((X_train_temp.shape[0], X_train_temp.shape[1],
                       X_train_temp.shape[2], 3), np.nan, dtype=np.float32)
    # Fill the 4D array with the 3 channels
    for i, element in enumerate(X_train_temp):
        for j, row in enumerate(element):
            for k, value in enumerate(row):
                # Check if the value is np.array([nan, nan, nan])
                if np.isnan(value).all():
                    X_train_temp[i][j][k] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                else:
                    X_train_temp[i][j][k] = value

                X_train[i][j][k] = X_train_temp[i][j][k][:3]

    variance_metric, _ = get_variance_of_metrics(X_train)

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[2]):
            X_train[i][:, j] = handle_fill_nan_by_variance(X_train[i][:, j], variance_metric[i][j])

    return X_train, y_train_temp
