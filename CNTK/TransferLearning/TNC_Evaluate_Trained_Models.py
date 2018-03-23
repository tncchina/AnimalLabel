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
import math
import sys
import glob

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


# Evaluates an image set using the provided model
def eval_test_images(loaded_model, output_file_test_predict, test_map_file, image_width, image_height, max_images=-1, column_offset=0):
    num_images = sum(1 for line in open(test_map_file))
    if max_images > 0:
        num_images = min(num_images, max_images)
    #print("Evaluating saved models output node for {0} images.".format(num_images))

    pred_count = 0
    correct_count = 0
    test_correct_rate = 0.0
    np.seterr(over='raise')

    with open(output_file_test_predict, 'w') as test_predict_file:
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
                test_predict_file.write("%s,%d,%d,%0.3f\n" % (os.path.basename(img_file), true_label, predicted_label, np.amax(probs)))

                #if pred_count % 100 == 0:
                #    print("Processed {0} samples ({1} correct)".format(pred_count, (float(correct_count) / pred_count)))
                if pred_count >= num_images:
                    break

    test_correct_rate = float(correct_count) / pred_count
    print("{0} out of {1} predictions were correct {2}.".format(correct_count, pred_count, test_correct_rate))
    return test_correct_rate


if __name__ == '__main__':
    # when calling this script, two command-line arguments should be passed in:
    #    1. the relative folder name of the output folder should be passed in as a command-line argument.  The trained model files should be stored in this folder.
    #    2. the file name of the test map file.
    base_folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_folder, sys.argv[1])
    test_map_file_full_path_name = os.path.join(output_folder, sys.argv[2])

    if not os.path.exists(output_folder):
        print(output_folder + " doesn't exist.")
        exit(0)

    if not os.path.exists(test_map_file_full_path_name):
        print(test_map_file_full_path_name + " doesn't exist.")
        exit(0)

    # define base model location and characteristics
    _image_height = 682
    _image_width = 512
    _num_channels = 3

    plot_data = {'epoch_index': list(), 'test_correct_rate': list()}
    epoch_index = 0
    # get a list of all *.model files (saved CNTK model files) in the output folder.
    model_file_full_path_names =  glob.glob(os.path.join(output_folder, "*.model"))

    for model_file_full_path_name in model_file_full_path_names:
        print("Evaluting " + model_file_full_path_name)
        model_base_file_name = os.path.basename(model_file_full_path_name)
        model_prediction_file_name = os.path.splitext(model_base_file_name)[0] + "_Predictions.txt"
        model_prediction_file_full_path_name = os.path.join(output_folder, model_prediction_file_name)
        #print(model_prediction_file_full_path_name)

        trained_model = load_model(model_file_full_path_name)

        # Evaluate model against the test set
        test_correct_rate = eval_test_images(trained_model, model_prediction_file_full_path_name, test_map_file_full_path_name, _image_width, _image_height)

        epoch_index += 1
        plot_data['epoch_index'].append(epoch_index)
        plot_data['test_correct_rate'].append(test_correct_rate)


    # Visualize the test correct rates of all saved models:
    window_width = 32
    test_correct_rate_cumsum = np.cumsum(np.insert(plot_data['test_correct_rate'], 0, 0))

    plt.figure(1)
    plt.plot(plot_data["epoch_index"], plot_data["test_correct_rate"], 'r--')
    plt.xlabel('Epoch number')
    plt.ylabel('Test Correct Rate')
    plt.title('Epoch run vs. Test Correct Rate ')
    #plt.show()
    output_figure_Test_Correct_Rate = os.path.join(output_folder, "Epoch_Test_Correct_Rate.png")
    plt.savefig(output_figure_Test_Correct_Rate, bbox_inches='tight')


    print("Done. All models have been evaluated.")
