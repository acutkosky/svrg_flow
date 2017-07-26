from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf
import numpy as np


#hacky subclass of Optimizer just so I can use make_slot...
#not sure if it's even that helpful over a plain dict of vars...
class SVRG(Optimizer):
    '''SVRG
    implements stochastic variance reduced gradient descent
    sgd_optimizer: optimizer to use for SGD phase of SVRG
    loss_fnc: function that takes as arguments a (list of) variables and outputs
        a tensor containing the loss.
    '''

    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def __init__(self, sgd_optimizer, loss_func, use_locking=False, name=None):
        '''
        constructs a new SVRG optimizer

        '''

        if name is None:
            name = 'SVRG_with_' + sgd_optimizer.get_name()


        super(SVRG, self).__init__(use_locking, name)

        self.sgd_optimizer = sgd_optimizer
        self.loss_func = loss_func
        self.computed_batch = False
        self.var_list = []
        self._slots = {}

    # def _get_or_make_slot(self, v, val, name):
    #     if v not in self._slots:
    #         self._slots[v] = {}
    #     self._slots[v][name] = val

    # def _get_slot(self, v, name):
    #     return self._slots[v][name]

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                batch_gradient = constant_op.constant(0,
                                              shape=v.get_shape(),
                                              dtype=v.dtype.base_dtype)
                v_prev = constant_op.constant(0.0,shape=v.get_shape(), dtype = v.dtype.base_dtype)
            self._get_or_make_slot(v, batch_gradient, "batch_gradient", self._name)
            self._get_or_make_slot(v, v_prev, "v_prev", self._name)


    def compute_gradients(self, var_list=None,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):

        if var_list is None:
            var_list = self.var_list

        assert self.computed_batch

        prev_vars = [self.get_slot(var, "v_prev") for var in var_list]
        prev_grads_and_vars = self.sgd_optimizer.compute_gradients(self.loss_func(prev_vars),
                                                              prev_vars,
                                                              gate_gradients,
                                                              aggregation_method,
                                                              colocate_gradients_with_ops,
                                                              grad_loss)

        grads_and_vars = self.sgd_optimizer.compute_gradients(self.loss_func(var_list),
                                                              var_list,
                                                              gate_gradients,
                                                              aggregation_method,
                                                              colocate_gradients_with_ops,
                                                              grad_loss)

        def reduce_variance(grad_var, prev_grad_var):

            (grad, var) = grad_var
            (prev_grad, prev_var) = prev_grad_var

            batch_gradient = self.get_slot(var, "batch_gradient")

            if grad is None or prev_grad is None or batch_gradient is None:
                return (None, var)
            return (grad - prev_grad + batch_gradient, var)

        return [reduce_variance(*pair) for pair in zip(grads_and_vars, prev_grads_and_vars)]


    def set_var_list(self, var_list):
        self.var_list = var_list

    def recompute_batch(self, batch_loss, var_list=None):
        self.computed_batch = True
        if var_list is None:
            var_list = self.var_list

        if not hasattr(var_list, '__iter__'):
            batch_gradients_and_vars = [var_list]

        self._create_slots(var_list)

        batch_gradients_and_vars = tf.gradients(batch_loss, var_list)

        if not hasattr(batch_gradients_and_vars, '__iter__'):
            batch_gradients_and_vars = [batch_gradients_and_vars]

        batch_grad_updates = [state_ops.assign(self.get_slot(var, "batch_gradient"), grad)
                                for grad, var in zip(batch_gradients_and_vars, var_list)]
        prev_vars_update = [state_ops.assign(self.get_slot(var, "v_prev"), var) for var in var_list]

        updates = batch_grad_updates + prev_vars_update

        return control_flow_ops.group(*updates)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        return self.sgd_optimizer.apply_gradients(grads_and_vars, global_step, name)

    def minimize(self, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):

        if var_list is None:
            var_list = self.var_list

        grads_and_vars = self.compute_gradients(var_list,
                                                gate_gradients,
                                                aggregation_method,
                                                colocate_gradients_with_ops,
                                                grad_loss)

        return self.apply_gradients(grads_and_vars, global_step, name)


