import tensorflow as tf
import os
import numpy as np
import abc
import layers as layers
from config import Config
def get(model_name, inputs, checkpoint_dir, is_training=False,reuse=False):
    return globals()[model_name](inputs, checkpoint_dir, is_training,reuse)
class cosac(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, inputs,checkpoint_dir,is_training=False,reuse=False):
        self.inputs = inputs
        self.checkpoint_dir = checkpoint_dir
        self.is_training = is_training
        self.summaries = []
        self.layers = {}
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self._prediction()
        self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=100)  
        self.config = Config


    '''def conv(self,inputs,n_filters,ksize,stride,scope=None, reuse=None, activation_fn=tf.nn.relu,
             initializer=tf.contrib.layers.variance_scaling_initializer(),
             padding='SAME'):

        with tf.variable_scope(scope,reuse=reuse):
            print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
            n_in = inputs.get_shape().as_list()[-1]
            print(n_in)

            weights = tf.get_variable(
              'weights',shape=[ksize,n_in,n_filters],
              initializer=initializer,
              #dtype = inputs.dytpes.base_dtype,
              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES])

            current_layer = tf.nn.conv1d(inputs, weights, stride, padding=padding)

            biases = tf.get_variable(
              'biases',shape=[n_filters,],
              initializer=tf.zeros_initializer(),
              #dtype = inputs.dytpes.base_dtype,
              collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])   

            current_layer = tf.nn.bias_add(current_layer, biases)
            current_layer = activation_fn(current_layer)      
            return current_layer'''

    def fcn(self,inputs,outputs,
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            regularizer=None, scope=None, reuse=None):

        with tf.variable_scope(scope,reuse=reuse):
            n_in = inputs.get_shape().as_list()[-1]
            print(n_in)
            print("**************************")
            weights = tf.get_variable('weights',
                                      shape=[n_in, outputs],
                                      dtype=inputs.dtype.base_dtype,
                                      initializer=initializer,
                                      regularizer=regularizer)
            current_layer = tf.matmul(inputs, weights)
            biases = tf.get_variable('biases',
                                     shape=[outputs,],
                                     dtype=inputs.dtype.base_dtype,
                                     initializer=tf.constant_initializer(0))
            current_layer = tf.nn.bias_add(current_layer, biases)
            return current_layer

    def _prediction(self):
        self.batch_size = self.inputs['data'].get_shape().as_list()[0]
        current_layer = self.inputs['data']
        n_filters = 20
        ksize = 8
        depth = 7

        for i in range(depth):
            current_layer = layers.conv(inputs=current_layer, n_filters=n_filters, ksize=ksize, stride=2, scope='conv{}'.format(i+1))
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
            self.layers['conv{}'.format(i+1)] = current_layer
            bs, width, _ = current_layer.get_shape().as_list()
            print(bs,width,_)
        bs, width, _ = current_layer.get_shape().as_list()
        print(bs,width,_)
        current_layer = tf.reshape(current_layer, [bs, width*n_filters])
        current_layer = self.fcn(current_layer,2,scope='logits')
        #current_layer = tf.nn.softmax(current_layer)
        self.layers['logits'] = current_layer
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)

        self.layers['class_prob'] = tf.nn.softmax(current_layer, name='class_prob')
        self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'],1,name='class_pred')
        tf.contrib.layers.apply_regularization(
          tf.contrib.layers.l2_regularizer(1e-3),
          weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
        #return self.layers['logits']
    def loss(self):
        with tf.name_scope('loss'):
            labels = self.inputs['label']
            logits = self.layers['logits']
        raw_loss = tf.reduce_mean(
           tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        self.loss = raw_loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_losses:
            with tf.name_scope('regularizers'):
                reg_loss = sum(reg_losses)
                self.summaries.append(tf.summary.scalar('loss/regularization', reg_loss))
            self.loss += reg_loss
        print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        print(self.loss)
        self.summaries.append(tf.summary.scalar('loss/train', self.loss))
    def optimizer(self,learning_rate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops, name='update_ops')
            with tf.control_dependencies([updates]):
                self.loss = tf.identity(self.loss)
        optim = tf.train.AdamOptimizer(learning_rate).minimize(
          self.loss, name='optimizer', global_step=self.global_step)
        self.optimizer = optim
    def _tofetch(self):
        return {
          'optimizer':self.optimizer,
          'loss':self.loss}
    def validation_metrics(self):
        if not hasattr(self, '_validation_metrics'):
            self.loss()

            self._validation_metrics = {
              'loss': self.loss,
              'pred': self.layers['class_prob']
               }
        return self._validation_metrics

    def validation_metrics_message(self, metrics):
        s = 'loss = {:.5f} | pred = {} '.format(metrics['loss'],
          metrics['pred'])
        return s
    def _train_step(self, sess, run_options=None, run_metadata=None):
        tofetch = self._tofetch()
        tofetch['step'] = self.global_step
        tofetch['summaries'] = self.merged_summaries
        data = sess.run(tofetch, options=run_options, run_metadata=run_metadata)
        return data
    def _test_step(self,sess,run_options=None,run_metadata=None):
        tofetch = self._tofetch()
        tofetch['step'] = self.global_step
        data = sess.run(tofetch, options=run_options, run_metadata=run_metadata)
        return data
    def _summary_step(self,step_data):
        step = step_data['step']
        loss = step_data['loss']
        if self.is_training:
            toprint = 'Step {} | loss = {:.4f}'.format(step, loss)
            f = open("loss.txt","a")
            print(np.float(loss),file=f)
        else:
            toprint = 'Step {} '.format(step)
        return toprint
    def save(self,sess):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(sess, checkpoint_path, global_step=self.global_step)
    def load(self,sess,step=None):
        if step==None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir,"model-"+str(step))

        self.saver.restore(sess, checkpoint_path)
        step = tf.train.global_step(sess, self.global_step)
        print ('Loaded model at step {} from snapshot {}.'.format(step, checkpoint_path))
    def test(self,n_val_step):
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.load(sess)
            print ('Starting prediction on testing set.')
            for step in range(n_val_steps):
                step_data = self._test_step(sess,None,None)
            print(self._prediction())
            coord.request_stop()
            coord.join(threads)
    def train(self, learning_rate, resume=False, summary_step=1,
            checkpoint_step=500,profiling=False):
        lr = tf.Variable(learning_rate, name='learning_rate',
          trainable=False,
          collections=[tf.GraphKeys.VARIABLES])
        self.summaries.append(tf.summary.scalar('learning_rate', lr))
        self.loss()
        self.optimizer(lr)
        if profiling:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None
        self.merged_summaries = tf.summary.merge(self.summaries)
        with tf.Session() as sess:
            self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)
            print ('Initializing all variables.')
            tf.initialize_local_variables().run()
            tf.initialize_all_variables().run()
            if resume:
                self.load(sess)
            print ('Starting data threads coordinator.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
            try:
                while not coord.should_stop():
                    step_data = self._train_step(sess, run_options, run_metadata)
                    step = step_data['step']
                    if step >= 0 and step % summary_step == 0:
                        if profiling:
                            self.summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
                            tl = timeline.Timeline(run_metadata.step_stats)
                            ctf = tl.generate_chrome_trace_format()
                            with open(os.path.join(self.checkpoint_dir, 'timeline.json'), 'w') as fid:
                                print ('Writing trace.')
                                fid.write(ctf)
                        print(self._summary_step(step_data))
                        self.summary_writer.add_summary(step_data['summaries'], global_step=step)
                    if checkpoint_step is not None and (
                      step > 0) and step % checkpoint_step == 0:
                        print ('Step {} | Saving checkpoint.'.format(step))
                        self.save(sess)
            except KeyboardInterrupt:
                print ('Interrupted training at step {}.'.format(step))
                self.save(sess)
            except tf.errors.OutOfRangeError:
                print ('Training completed at step {}.'.format(step))
                self.save(sess)

            finally:
                print ('Shutting down data threads.')
                coord.request_stop()
                self.summary_writer.close()

            # Wait for data threads
            print ('Waiting for all threads.')
            coord.join(threads)

            print ('Optimization done.')
