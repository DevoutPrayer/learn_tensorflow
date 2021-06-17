from sklearn import datasets
import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

