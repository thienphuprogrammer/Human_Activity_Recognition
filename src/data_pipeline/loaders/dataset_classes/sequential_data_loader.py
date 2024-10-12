import tensorflow as tf


def load_data(file_path: str):
    data = tf.data.experimental.make_csv_dataset(file_path, batch_size=32)
    return data