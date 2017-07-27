import numpy as np
import tensorflow as tf
import svrg_flow

w = tf.Variable(0.0)
loss_func = lambda v: tf.nn.l2_loss(v[0] - 3)
loss = loss_func([w])

adagrad = tf.train.AdagradOptimizer(1.0)

svrg = svrg_flow.SVRG(adagrad, loss_func)
svrg.set_var_list([w])

batch_op = svrg.recompute_batch(loss_func)

min_op = svrg.minimize()

init = tf.global_variables_initializer()

s = tf.Session()
s.run(init)
s.run(batch_op)
print s.run(loss)
for t in xrange(100):
	s.run(min_op)

print s.run(loss)