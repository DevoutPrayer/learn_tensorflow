# 鸢尾花分类 
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 1.  准备数据
# 1.1 数据集读入
# 花萼长 花萼宽 花瓣长 花瓣宽
x_data = datasets.load_iris().data
# 
y_data = datasets.load_iris().target

# 1.2 数据集乱序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
# 1.3 生成测试集和训练集
x_train = x_data[:-30,:]
y_train = y_data[:-30]
x_test  = x_data[-30:,:]
y_test  = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test  = tf.cast(x_test, tf.float32)

# 1.4 配对
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(24)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(24)

# 2.  搭建网络
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))


# 3.  参数优化
lr = 0.1
epoch = 500
test_acc = []
loss_all = 0

for epoch in range(epoch):
        for step, (x_train, y_train) in enumerate(train_db):
                with tf.GradientTape() as tape:
                        y  = tf.matmul( x_train, w1) + b1
                        y  = tf.nn.softmax(y)
                        y_ = tf.one_hot(y_train, depth=3)
                        loss = tf.reduce_mean(tf.square(y_ - y))
                        loss_all += loss.numpy()
                grads = tape.gradient(loss, [w1, b1])
                w1.assign_sub(lr*grads[0])
                b1.assign_sub(lr*grads[1])
        print("Epoch {}, loss {}".format(epoch, loss_all))
        loss_all = 0
# 4.  测试效果
        total_correct, total_number = 0, 0
        for x_test, y_test in test_db:
                y = tf.matmul(x_test, w1) + b1
                y = tf.nn.softmax(y)
                pred = tf.argmax(y,axis=1)
                pred = tf.cast(pred, dtype=y_test.dtype)
                correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
                correct = tf.reduce_sum(correct)
                total_correct += int(correct)
                total_number += x_test.shape[0]
                acc = total_correct/total_number
                print("Test_acc", acc)
                print("-----------------------------")
