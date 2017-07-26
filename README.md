# svrg_flow
SVRG in tensorflow

A currently very basic implementation of SVRG. Usage:
```
w = tf.Variable(0.0)
loss_func = lambda v: tf.nn.l2_loss(v[0] - 10.0)

my_favorite_sgd = tf.train.GradientDescentOptimizer(0.1)
svrg = SVRG(my_favorite_sgd, loss_func)
batch_op = svrg.recompute_batch(loss_func([w]))
min_op = svrg.minimize()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(batch_op)
for step in xrange(100):
    sess.run(min_op)
```
