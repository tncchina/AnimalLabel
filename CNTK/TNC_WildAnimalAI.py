# https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_201B_CIFAR-10_ImageHandsOn.ipynb
# https://microsoft.github.io/CNTK-R//articles/cifar10_example.html
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import PIL
import sys
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

import cntk as C

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

data_path = os.path.join('C:\TNC_CNTK\data')

print("data_path: ", data_path)

# model dimensions
image_height = 3000
image_width = 4000
num_channels = 3
num_classes = 10

import cntk.io.transforms as xforms


#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train):
    print("Reading map file:", map_file)
    print("Reading mean file:", mean_file)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("Can not find map file or mean file.")

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    # train uses data augmentation (translation only)
    #if train:
    #    transforms += [
    #        xforms.crop(crop_type='randomside', side_ratio=0.8)
    #    ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        #xforms.mean(mean_file)
    ]

    # deserializer
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features=C.io.StreamDef(field='image', transforms=transforms),
        # first column in map file is referred to as 'image'
        labels=C.io.StreamDef(field='label', shape=num_classes)  # and second as 'label'
    )))

# Create the train and test readers
reader_train = create_reader(os.path.join(data_path, 'train_map.txt'),
                             os.path.join(data_path, 'TNC_mean.xml'), True)
reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'),
                             os.path.join(data_path, 'TNC_mean.xml'), False)


def create_basic_model(input, out_dims):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        net = C.layers.Convolution((5, 5), 32, pad=True)(input)
        net = C.layers.MaxPooling((32, 32), strides=(8, 8))(net)

        net = C.layers.Convolution((5, 5), 32, pad=True)(net)
        net = C.layers.MaxPooling((3, 3), strides=(2, 2))(net)

        net = C.layers.Convolution((5, 5), 64, pad=True)(net)
        net = C.layers.MaxPooling((3, 3), strides=(2, 2))(net)

        net = C.layers.Dense(64)(net)
        net = C.layers.Dense(out_dims, activation=None)(net)

    return net


#
# Train and evaluate the network.
#
def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):

    print("train_and_evaluate 1")
    print(datetime.datetime.now())


    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    print("train_and_evaluate 2")

    print(datetime.datetime.now())

    # Normalize the input
    feature_scale = 1.0 / 256.0
    input_var_norm = C.element_times(feature_scale, input_var)

    print("train_and_evaluate 3")

    print(datetime.datetime.now())

    # apply model to input
    z = model_func(input_var_norm, out_dims=10)

    print("train_and_evaluate 4")

    print(datetime.datetime.now())

    #
    # Training action
    #

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    print("train_and_evaluate 5")

    print(datetime.datetime.now())
    # training config
    #epoch_size = 50000
    #minibatch_size = 64
    epoch_size = 10
    minibatch_size = 1


    # Set training parameters
    lr_per_minibatch = C.learning_rate_schedule([0.01] * 10 + [0.003] * 10 + [0.001],
                                                C.UnitType.minibatch, epoch_size)
    momentum_time_constant = C.momentum_as_time_constant_schedule(-minibatch_size / np.log(0.9))
    l2_reg_weight = 0.001

    print("train_and_evaluate 5.1")

    print(datetime.datetime.now())
    # trainer object
    learner = C.momentum_sgd(z.parameters,
                             lr=lr_per_minibatch,
                             momentum=momentum_time_constant,
                             l2_regularization_weight=l2_reg_weight)

    print("train_and_evaluate 5.2")

    print(datetime.datetime.now())

    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), [learner], [progress_printer])

    print("train_and_evaluate 6")

    print(datetime.datetime.now())


    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    C.logging.log_number_of_parameters(z);
    print()

    # perform model training

    print("train_and_evaluate 7")

    print(datetime.datetime.now())
    batch_index = 0
    plot_data = {'batchindex': [], 'loss': [], 'error': []}
    for epoch in range(max_epochs):  # loop over epochs

        print("train_and_evaluate 8: epoch: ", epoch)

        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            print("train_and_evaluate 8.1: sample_count", sample_count)

            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                               input_map=input_map)  # fetch minibatch.
            trainer.train_minibatch(data)  # update model with it

            sample_count += data[label_var].num_samples  # count samples processed so far

            # For visualization...
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        trainer.summarize_training_progress()

    print("train_and_evaluate 9")

    print(datetime.datetime.now())

    #
    # Evaluation action
    #
    epoch_size = 10
    minibatch_size = 1

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index + 1,
                                                                        (metric_numer * 100.0) / metric_denom,
                                                                        metric_denom))
    print(datetime.datetime.now())
    print("")

    # Visualize training result:
    window_width = 32
    loss_cumsum = np.cumsum(np.insert(plot_data['loss'], 0, 0))
    error_cumsum = np.cumsum(np.insert(plot_data['error'], 0, 0))

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss'] = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error'] = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width

    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')

    plt.show()

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.show()

    return C.softmax(z)

pred = train_and_evaluate(reader_train,
                          reader_test,
                          max_epochs=2,
                          model_func=create_basic_model)


