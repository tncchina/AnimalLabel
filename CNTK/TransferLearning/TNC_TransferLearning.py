# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import cntk as C
import os
from PIL import Image
from cntk.device import try_set_default_device, gpu
from cntk import load_model, placeholder, Constant
from cntk import Trainer
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk.layers import Dense
from cntk.learners import momentum_sgd, learning_parameter_schedule, momentum_schedule
from cntk.ops import combine, softmax
from cntk.ops.functions import CloneMethod
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import log_number_of_parameters, ProgressPrinter

import matplotlib.pyplot as plt
import datetime
import shutil

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ConfusionMatrix import ConfusionMatrix

################################################
################################################
# general settings
make_mode = False
freeze_weights = False
base_folder = os.path.dirname(os.path.abspath(__file__))
start_datetime = datetime.datetime.now()

# create the output folder
data_source_folder = os.path.join(base_folder, "Output")
output_folder = os.path.join(base_folder, "Output_" + start_datetime.strftime("%Y%m%d%H%M"))
#if not os.path.exists(output_folder):     # create Output folder to store our output files.
#    os.makedirs(output_folder)
print("Copying ", data_source_folder, " to ", output_folder)
shutil.copytree(data_source_folder, output_folder )

output_file = os.path.join(output_folder, "PredictionsOutput.txt")
output_file_test_predict = os.path.join(output_folder, "Test_Prediction.txt")
output_figure_loss = os.path.join(output_folder, "Training_Loss.png")
output_figure_error = os.path.join(output_folder, "Training_Prediction_Error.png")
output_figure_validation_correct_rate = os.path.join(output_folder, "Training_validation_correct_rate.png")

confusion_matrix_file = os.path.join(output_folder, "ConfustionMatrix.txt")
validation_correct_rate_file = os.path.join(output_folder, "Validation_Correct_Rate.txt")

log_file_name = os.path.join(output_folder, "Logs.txt")

features_stream_name = 'features'
label_stream_name = 'labels'
new_output_node_name = "prediction"

# Learning parameters
max_epochs = 80
mb_size = 25
lr_per_mb = [0.1]*30 + [0.01]*30 + [0.001]*30
momentum_per_mb = 0.9
l2_reg_weight = 0.00005

# define base model location and characteristics
_base_model_name = "ResNet18_ImageNet_CNTK.model"
#_base_model_name = "VGG16_ImageNet_Caffe.model"
_base_model_file = os.path.join(base_folder, "PretrainedModels", _base_model_name)
_feature_node_name = "features"
_last_hidden_node_name = "z.x"
_image_width = 682
_image_height = 512
_num_channels = 3

# define the file name we will save our trained model to.  It is "TNC_" + _base_model_name
tl_model_file = os.path.join(output_folder, "TNC_" + _base_model_name)

# define data location and characteristics
_data_folder = os.path.join(base_folder, "DataSets")
_train_map_filename = "TNC512_train_random.txt"
_test_map_filename = "TNC512_test_random.txt"
_train_map_file = os.path.join(output_folder, _train_map_filename)
_test_map_file = os.path.join(output_folder,  _test_map_filename)

# get the number of classes from training set
_num_classes = 0
with open(_train_map_file, 'r') as train_file:
    for line in train_file:
        file_class_id = line.split('\t')
        class_id = int(file_class_id[1])
        if class_id>_num_classes:
            _num_classes = class_id
    train_file.close()

_num_classes += 1

# Log basic configuration to Output\Configuration.txt
_base_model_ID_file_name = os.path.join(output_folder, "Configuration.txt")
with open(_base_model_ID_file_name, 'w') as base_model_id_file:
    base_model_id_file.write("Base Model: %s\n" % _base_model_name)
    base_model_id_file.write("Feature node name: %s\n" % _feature_node_name)
    base_model_id_file.write("Last hidden node: %s\n" % _last_hidden_node_name)
    base_model_id_file.write("Image height  : %d\n" % _image_height)
    base_model_id_file.write("Image width   : %d\n" % _image_width)
    base_model_id_file.write("Image channels: %d\n" % _num_channels)
    base_model_id_file.write("Training set: %s\n" % _train_map_file)
    base_model_id_file.write("Test set    : %s\n" % _test_map_file)
    base_model_id_file.write("Number of classes: %d\n" % _num_classes)
    base_model_id_file.write("Training Parameters:\n")
    base_model_id_file.write("  Max epochs = %d\n" % max_epochs)
    base_model_id_file.write("  Mini-batch size = %d\n" % mb_size)
    base_model_id_file.write("  Learning rate/mb = %s\n" % str(lr_per_mb))
    base_model_id_file.write("  Momentum/mb = %f\n" % momentum_per_mb)
    base_model_id_file.write("  L2 regression weight = %f\n" % l2_reg_weight)

################################################
################################################


# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_width, image_height, num_channels, num_classes, randomize=True):
    transforms = []
    transforms += [xforms.crop(crop_type='randomside', side_ratio=0.8)]
    transforms += [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
            features =StreamDef(field='image', transforms=transforms),
            labels   =StreamDef(field='label', shape=num_classes))),
            randomize=randomize)


# Creates the network model for transfer learning
def create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, input_features, freeze=False):
    # Load the pretrained classification net and find nodes
    base_model   = load_model(base_model_file)
    feature_node = find_by_name(base_model, feature_node_name)
    last_node    = find_by_name(base_model, last_hidden_node_name)

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Add new dense layer for class prediction
    feat_norm  = input_features - Constant(114)
    feat_norm  = C.element_times(1.0/256.0, feat_norm)
    cloned_out = cloned_layers(feat_norm)
    z          = Dense(num_classes, activation=None, name=new_output_node_name) (cloned_out)

    return z


# Evaluates a single image using the provided model
def eval_single_image(loaded_model, image_path, image_width, image_height):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    img = Image.open(image_path)
    if image_path.endswith("png"):
        temp = Image.new("RGB", img.size, (255, 255, 255))
        temp.paste(img, img)
        img = temp
    resized = img.resize((image_width, image_height), Image.ANTIALIAS)
    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    ## Alternatively: if you want to use opencv-python
    # cv_img = cv2.imread(image_path)
    # resized = cv2.resize(cv_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    # bgr_image = np.asarray(resized, dtype=np.float32)
    # hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    # compute model output
    arguments = {loaded_model.arguments[0]: [hwc_format]}
    output = loaded_model.eval(arguments)

    # return softmax probabilities
    sm = softmax(output[0])
    return sm.eval()


# Trains a transfer learning model
def train_model(base_model_file, feature_node_name, last_hidden_node_name,
                image_width, image_height, num_channels, num_classes, train_map_file,
                num_epochs, max_images=-1, freeze=False):
    epoch_size = sum(1 for line in open(train_map_file))
    if max_images > 0:
        epoch_size = min(epoch_size, max_images)

    # Create the mini-batch source and input variables
    minibatch_source = create_mb_source(train_map_file, image_width, image_height, num_channels, num_classes)
    image_input = C.input_variable((num_channels, image_height, image_width))
    label_input = C.input_variable(num_classes)

    # Define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[features_stream_name],
        label_input: minibatch_source[label_stream_name]
    }

    # Instantiate the transfer learning model and loss function
    tl_model = create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, image_input, freeze)
    ce = cross_entropy_with_softmax(tl_model, label_input)
    pe = classification_error(tl_model, label_input)

    # Instantiate the trainer object
    lr_schedule = learning_parameter_schedule(lr_per_mb)
    mm_schedule = momentum_schedule(momentum_per_mb)
    learner = momentum_sgd(tl_model.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    progress_printer = ProgressPrinter(tag='Training', log_to_file=log_file_name, num_epochs=num_epochs)
    #progress_printer = ProgressPrinter(tag='Training', log_to_file=log_file_name, num_epochs=num_epochs)
    trainer = Trainer(tl_model, (ce, pe), learner, progress_printer)

    # Get mini-batches of images and perform model training
    print("Training transfer learning model for {0} epochs (epoch_size = {1}).".format(num_epochs, epoch_size))
    batch_index = 0
    plot_data = {'batchindex': list(), 'loss': list(), 'error': list(),
                 'epoch_index':list(), 'validation_correct_rate': list()}
    log_number_of_parameters(tl_model)
    
    vcr_log = open(validation_correct_rate_file, 'w')

    for epoch in range(num_epochs):       # loop over epochs
        print("")
        print("===== EPOCH {0} =======".format(epoch))
        sample_count = 0
        while sample_count < epoch_size:  # loop over mini-batches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            print("Epoch {0}: Processed {1} of {2} samples".format(epoch, sample_count, epoch_size))
            #if sample_count % (100 * mb_size) == 0:
            #    print ("Processed {0} samples".format(sample_count))
            # For visualization...
            #print("type of plot_data:", type(plot_data), type(plot_data['batchindex']), type(plot_data['loss']),type(plot_data['error']))
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        # Evaluate the model on the validation
        validation_correct_rate = eval_validation_images_during_training(tl_model, output_file, _test_map_file,
                                                                         _image_width, _image_height)
        plot_data['epoch_index'].append(epoch)
        plot_data['validation_correct_rate'].append(validation_correct_rate)

        vcr_log.write("{0}\t{1}\n".format(epoch+1, validation_correct_rate))

        # for every epoch, save the trained model.
        model_file_for_current_epoch = os.path.join(output_folder, "TNC_ResNet18_ImageNet_CNTK_" + ("%03d" % epoch) + ".model")
        tl_model.save(model_file_for_current_epoch)

        trainer.summarize_training_progress()

    vcr_log.flush()
    vcr_log.close()

    # Visualize training result:
    window_width = 32
    loss_cumsum = np.cumsum(np.insert(plot_data['loss'], 0, 0))
    error_cumsum = np.cumsum(np.insert(plot_data['error'], 0, 0))

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss'] = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error'] = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width

    plt.figure(1)
    #plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Mini-batch number')
    plt.ylabel('Loss')
    plt.title('Mini-batch run vs. Training loss ')
    #plt.show()
    plt.savefig(output_figure_loss, bbox_inches='tight' )

    plt.figure(2)
    #plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Mini-batch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Mini-batch run vs. Training Prediction Error ')
    #plt.show()
    plt.savefig(output_figure_error, bbox_inches='tight')

    plt.figure(3)
    #plt.subplot(212)
    plt.plot(plot_data["epoch_index"], plot_data["validation_correct_rate"], 'r--')
    plt.xlabel('Epoch number')
    plt.ylabel('Validation Correct Rate')
    plt.title('Epoch vs. Validation Correct Rate ')
    #plt.show()
    plt.savefig(output_figure_validation_correct_rate, bbox_inches='tight')

    return tl_model



# Evaluates an image set using the provided model
def eval_test_images(loaded_model, output_file, test_map_file, image_width, image_height, max_images=-1, column_offset=0):
    num_images = sum(1 for line in open(test_map_file))
    if max_images > 0:
        num_images = min(num_images, max_images)
    print("Evaluating model output node '{0}' for {1} images.".format(new_output_node_name, num_images))

    pred_count = 0
    correct_count = 0
    test_correct_rate = 0.0
    np.seterr(over='raise')

    cm = ConfusionMatrix()

    with open(output_file, 'wb') as results_file, open(output_file_test_predict, 'w') as test_predict_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0 + column_offset]
                probs = eval_single_image(loaded_model, img_file, image_width, image_height)

                pred_count += 1
                true_label = int(tokens[1 + column_offset])
                predicted_label = np.argmax(probs)
                if predicted_label == true_label:
                    correct_count += 1
                np.savetxt(results_file, probs[np.newaxis], fmt="%.3f", delimiter=',', newline='\n')
                #np.savetxt(confusion_matrix_file, (true_label, predicted_label, np.amax(probs)), fmt="%d %d %.3f", delimiter=',', newline='\n')
                #np.savetxt(confusion_matrix_file, (true_label, predicted_label), fmt="%d %d",  delimiter=',', newline='\n')
                #csv_writer.writerow([true_label, predicted_label])
                test_predict_file.write("%s,%d,%d,%0.3f\n" % (os.path.basename(img_file), true_label, predicted_label, np.amax(probs)))
                cm.add_result(int(true_label), int(predicted_label))

                if pred_count % 100 == 0:
                    print("Processed {0} samples ({1} correct)".format(pred_count, (float(correct_count) / pred_count)))
                if pred_count >= num_images:
                    break

    cm.set_id_lookup_file(os.path.join(output_folder,"Label_ClassID_Lookup.csv"))
    cm.change_id_to_label()
    cm.print_matrix()
    cm.savetxt(confusion_matrix_file)

    test_correct_rate = float(correct_count) / pred_count
    print("{0} out of {1} predictions were correct {2}.".format(correct_count, pred_count, test_correct_rate))




# Evaluates an image set using the provided model
def eval_validation_images_during_training(loaded_model, output_file, test_map_file, image_width, image_height, max_images=-1, column_offset=0):
    num_images = sum(1 for line in open(test_map_file))
    if max_images > 0:
        num_images = min(num_images, max_images)
    print("Evaluating model output node '{0}' for {1} images.".format(new_output_node_name, num_images))

    pred_count = 0
    correct_count = 0
    validation_correct_rate = 0.0
    np.seterr(over='raise')

    with open(test_map_file, "r") as input_file:
        for line in input_file:
            tokens = line.rstrip().split('\t')
            img_file = tokens[0 + column_offset]
            probs = eval_single_image(loaded_model, img_file, image_width, image_height)

            pred_count += 1
            true_label = int(tokens[1 + column_offset])
            predicted_label = np.argmax(probs)
            if predicted_label == true_label:
                correct_count += 1

            if pred_count >= num_images:
                break

    validation_correct_rate = float(correct_count) / pred_count
    print("{0} out of {1} predictions were correct {2}.".format(correct_count, pred_count, validation_correct_rate))
    return validation_correct_rate


# Evaluates an image set using the provided model
def eval_validation_images(loaded_model, output_file, test_map_file, image_width, image_height, max_images=-1, column_offset=0):
    num_images = sum(1 for line in open(test_map_file))
    if max_images > 0:
        num_images = min(num_images, max_images)
    print("Evaluating model output node '{0}' for {1} images.".format(new_output_node_name, num_images))

    pred_count = 0
    correct_count = 0
    validation_correct_rate = 0.0
    np.seterr(over='raise')

    cm = ConfusionMatrix()

    with open(output_file, 'wb') as results_file, open(output_file_test_predict, 'w') as test_predict_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0 + column_offset]
                probs = eval_single_image(loaded_model, img_file, image_width, image_height)

                pred_count += 1
                true_label = int(tokens[1 + column_offset])
                predicted_label = np.argmax(probs)
                if predicted_label == true_label:
                    correct_count += 1
                np.savetxt(results_file, probs[np.newaxis], fmt="%.3f", delimiter=',', newline='\n')
                #np.savetxt(confusion_matrix_file, (true_label, predicted_label, np.amax(probs)), fmt="%d %d %.3f", delimiter=',', newline='\n')
                #np.savetxt(confusion_matrix_file, (true_label, predicted_label), fmt="%d %d",  delimiter=',', newline='\n')
                #csv_writer.writerow([true_label, predicted_label])
                test_predict_file.write("%s,%d,%d,%0.3f\n" % (os.path.basename(img_file), true_label, predicted_label, np.amax(probs)))
                cm.add_result(int(true_label), int(predicted_label))

                if pred_count % 100 == 0:
                    print("Processed {0} samples ({1} correct)".format(pred_count, (float(correct_count) / pred_count)))
                if pred_count >= num_images:
                    break

    cm.set_id_lookup_file(os.path.join(output_folder,"Label_ClassID_Lookup.csv"))
    cm.change_id_to_label()
    cm.print_matrix()
    cm.savetxt(confusion_matrix_file)

    validation_correct_rate = float(correct_count) / pred_count
    print("{0} out of {1} predictions were correct {2}.".format(correct_count, pred_count, validation_correct_rate))




if __name__ == '__main__':
    try_set_default_device(gpu(0))
    # check for model and data existence
    if not (os.path.exists(_base_model_file) and os.path.exists(_train_map_file) and os.path.exists(_test_map_file)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # You can use the following to inspect the base model and determine the desired node names
    # node_outputs = get_node_outputs(load_model(_base_model_file))
    # for out in node_outputs: print("{0} {1}".format(out.name, out.shape))

    # Train only if no model exists yet or if make_mode is set to False
    if os.path.exists(tl_model_file) and make_mode:
        print("Loading existing model from %s" % tl_model_file)
        trained_model = load_model(tl_model_file)
    else:
        trained_model = train_model(_base_model_file, _feature_node_name, _last_hidden_node_name,
                                    _image_width, _image_height, _num_channels, _num_classes, _train_map_file,
                                    max_epochs, freeze=freeze_weights)
        trained_model.save(tl_model_file)
        print("Stored trained model at %s" % tl_model_file)

    # Evaluate the test set
    eval_test_images(trained_model, output_file, _test_map_file, _image_width, _image_height)

    print("Done. Wrote output to %s" % output_file)
