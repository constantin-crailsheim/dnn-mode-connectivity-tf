import tensorflow as tf


def pytest_configure(config):
    tf.random.set_seed(1)
