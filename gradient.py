import tensorflow as tf
 
x=tf.Variable(initial_value=[[1.,2.,3.]])
 
with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x + x
    dy_dx = gg.gradient(y, x)     # 求一阶导数
d2y_dx2 = g.gradient(dy_dx, x)    # 求二阶导数
 
print(y)
print(dy_dx)
print(d2y_dx2)
