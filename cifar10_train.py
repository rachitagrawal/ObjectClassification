import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
import cifar10
import numpy as np


cifar10.data_path = "./data/"
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


def pre_process_image(image, training):
    if training:
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        
        image = tf.image.random_flip_left_right(image)
        
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        
        #Make sure the pixel values are between [0, 1]
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                      target_height=img_size_cropped,
                                                      target_width=img_size_cropped)
        
    return image

def pre_process(images, training):
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    
    return images

def main_network(images, training):
    x_pretty = pt.wrap(images)
    
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer
        
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)
        
        return y_pred, loss

def create_network(training):
    with tf.variable_scope('network', reuse=not training):
        images = x
        images = pre_process(images=images, training=training)
        
        y_pred, loss = main_network(images=images, training=training)
        
    return y_pred, loss




global_step = tf.Variable(initial_value=0,
                         name='global_step', trainable=False)

_, loss = create_network(training=True)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)



y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



saver = tf.train.Saver()



def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')



def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Relu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor




output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')




session = tf.Session()

save_dir = 'checkpoints/'



if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')


try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())


train_batch_size = 64

def random_batch():
    num_images = len(images_train)
    
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]
    
    return x_batch, y_batch

def optimize(num_iterations):
    start_time = time.time()
    
    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        i_global, _ = session.run([global_step, optimizer],
                                 feed_dict=feed_dict_train)
        
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))
            
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            saver.save(session,
                      save_path=save_path,
                      global_step=global_step)
            print("Saved checkpoint")
            
        end_time = time.time()
        time_diff = end_time - start_time
        
        #print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))


session = tf.Session()
save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    print("Trying to restor last checkpoint....")
    
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    
    saver.restore(session, save_path=last_chk_path)
    
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())



optimize(num_iterations=25000)

