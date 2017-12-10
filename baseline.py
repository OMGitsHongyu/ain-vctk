import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from analyzer import *
import time
import pdb

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def create_error_metrics(gen, inputs, origs):
    # Losses

    # metric: L2 between downsampled generated output and input
    gen_LR = slim.avg_pool2d(gen, [4, 4], stride=1, padding='SAME')
    gen_mse_LR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_LR - inputs)), 1)
    gen_L2_LR = tf.reduce_mean(gen_mse_LR)

    # metric: L2 between generated output and the original image
    gen_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen - origs)), 1)
    # average for the batch
    gen_L2_HR = tf.reduce_mean(gen_mse_HR)

    # metric: PSNR between generated output and original input
    gen_rmse_HR = tf.sqrt(gen_mse_HR)
    gen_PSNR = tf.reduce_mean(20*tf.log(1.0/gen_rmse_HR)/tf.log(tf.constant(10, dtype=tf.float32)))

    err_im_HR = gen - origs
    err_im_LR = gen_LR - inputs

    return gen_L2_LR, gen_L2_HR, gen_PSNR, err_im_LR, err_im_HR


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True, b_reuse=False):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                # work around reuse=True problem
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                    self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                    with tf.control_dependencies([ema_apply_op]):
                        mean, var = tf.identity(batch_mean), tf.identity(batch_var)

        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed



########
# CONFIG 
#########
adam_learning_rate = 0.0001
adam_beta1 = 0.9

batch_size = 8
image_h = 256
image_w = 513

num_epochs = 20

###############################
# BUILDING THE MODEL
###############################


real_ims = tf.placeholder(tf.float32, [None, image_h, image_w, 1], name='real_ims')
inputs = tf.placeholder(tf.float32, [None, image_h, image_w, 1], name='inputs')


# generator section
print "GENERATOR"
print "-----------"

# not great way to create these
batch_norm_list = []
nb_residual = 4
n_extra_bn = 1
for n in range(nb_residual*2 + n_extra_bn):
    batch_norm_list.append(batch_norm(name='bn'+str(n)))

def create_generator(inputs, b_training=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = inputs
        print_shape(net)

        net = tf.nn.relu(slim.conv2d(net, 16, [3, 3], scope='gconv1'))
        print_shape(net)

        net1 = net

        res_inputs = net1
        for n in range(nb_residual):
            net = tf.nn.relu(batch_norm_list[n*2](slim.conv2d(res_inputs, 16, [3, 3], scope='conv1_res'+str(n)), train=b_training))
            net = batch_norm_list[n*2+1](slim.conv2d(net, 16, [3, 3], scope='conv2_res'+str(n)), train=b_training)
            net = net + res_inputs
            res_inputs = net


        print_shape(net)

        net = batch_norm_list[-1](slim.conv2d(net, 16, [3, 3], scope='gconv2'), train=b_training) + net1
        print_shape(net)

        # deconv
        net = tf.nn.relu(slim.conv2d_transpose(net, 54, [5, 5], stride=1, scope='deconv1'))
        print_shape(net)

        net = tf.nn.relu(slim.conv2d_transpose(net, 54, [5, 5], stride=1, scope='deconv2'))
        print_shape(net)


        # tanh since images have range [-1,1]
        net = slim.conv2d(net, 1, [3, 3], scope='gconv3', activation_fn=tf.nn.tanh)
        print_shape(net)

    return net

with tf.variable_scope("generator") as scope:
    gen = create_generator(inputs)
    scope.reuse_variables()
    gen_test = create_generator(inputs, False)


gen_L2_LR, gen_L2_HR, gen_PSNR, err_im_LR, err_im_HR = create_error_metrics(gen, inputs, real_ims)

# metrics for testing stream
gen_L2_LR_t, gen_L2_HR_t, gen_PSNR_t, err_im_LR_t, err_im_HR_t = create_error_metrics(gen_test, inputs, real_ims)

# baselines: L2 and PSNR between bicubic upsampled input and original image

train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimize the generator and discriminator separately
g_loss = gen_L2_HR

g_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(g_loss, var_list=g_vars)
    
weight_saver = tf.train.Saver(max_to_keep=1)


# logging
g_L2HR_train     = tf.summary.scalar("gen_L2_HR", gen_L2_HR)
g_L2HR_test      = tf.summary.scalar("gen_L2_HR", gen_L2_HR_t)


merged_summary_train = tf.summary.merge([g_L2HR_train])
merged_summary_test = tf.summary.merge([g_L2HR_test])

# tf will automatically create the log directory
train_writer = tf.summary.FileWriter('./logs_pair_supervised/train')
test_writer = tf.summary.FileWriter('./logs_pair_supervised/test')

print "initialization done"


#############
# TRAINING
############

data_dir = 'data/matrix_sample/225/pair/'
data_train = glob.glob(data_dir+"/train/*.bin")
data_test = glob.glob(data_dir+"/test/*.bin")

print "data train:", len(data_train)
print "data test:", len(data_test)


# create directories to save checkpoint and samples
samples_dir = 'samples_pair_supervised'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'checkpoint_pair_supervised'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

b_load = False
ckpt_dir = 'data/matrix_sample/225/checkpoint_pair_supervised/model-60002'


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    if b_load:
        # optimistic_restore(sess, ckpt_dir)
        # print "successfully restored!"

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        weight_saver.restore(sess, ckpt.model_checkpoint_path)
        counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
        print "successfully restored!" + " counter:", counter
        
    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            
            batch_inputs, batch_origs = read_pair_batch_numpy(batch_filenames)
            
            # update networks
            
            fetches = [g_loss, g_optim]
            errG, _ = sess.run(fetches, feed_dict={inputs:batch_inputs, real_ims:batch_origs})
            

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errG))

#             if np.mod(counter, 30) == 1:

#                 # training metrics first
#                 train_summary = sess.run([merged_summary_train], feed_dict={inputs: batch_inputs, real_ims: batch_origs})
#                 train_writer.add_summary(train_summary[0], counter)

#                 # now testing metrics
#                 rand_idx = np.random.randint(len(data_test)-batch_size+1)
#                 sample_origs, sample_inputs = get_images(data_test[rand_idx: rand_idx+batch_size])

#                 sample = sess.run([gen_test], feed_dict={inputs: sample_inputs})

#                 err_im_HR = sess.run([err_im_HR_t], feed_dict={inputs: sample_inputs, real_ims: sample_origs})

#                 test_summary = sess.run([merged_summary_test], feed_dict={ inputs: sample_inputs, real_ims: sample_origs})
#                 test_writer.add_summary(test_summary[0], counter)

#                 # save an image, with the original next to the generated one
#                 resz_input = sample_inputs[0].repeat(axis=0,repeats=4).repeat(axis=1,repeats=4)
#                 merge_im = np.zeros( (image_h, image_h*4, 3) )
#                 merge_im[:, :image_h, :] = (sample_origs[0]+1)*127.5
#                 merge_im[:, image_h:image_h*2, :] = (resz_input+1)*127.5
#                 merge_im[:, image_h*2:image_h*3, :] = (sample[0][0]+1)*127.5
#                 merge_im[:, image_h*3:, :] = (err_im_HR[0][0]+1)*127.5
#                 imsave(samples_dir + '/test_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)



#             if np.mod(counter, 1000) == 2:
#                 weight_saver.save(sess, checkpoint_dir + '/model', counter)
#                 print "saving a checkpoint"

