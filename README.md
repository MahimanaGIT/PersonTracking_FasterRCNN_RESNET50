# Object Detection On Tensorflow

Using transfer learning on SSD MobileNet v2 to train Object Detection on custom Dataset

The steps are inspired by the article "https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d"

The model zoo from which the model was downloaded is:
    "https://github.com/tensorflow/models/blob/5245161c96cc057dc7a883ef4283ed7fab735bcf/research/object_detection/g3doc/tf1_detection_zoo.md"

Step 1: Saving custom images for the dataset

Step 2: Saving corresponding xml files for every images of the dataset using LabelImg

Step 3: Saving Images in data/images folder and separate corresponding xml files in "test_labels" and "training_labels" folder in ./data folder

Step 4: Converting corresponding xml files for "test_labels" and "training_labels" to CSV files and label_map.pbtxt file using the script "Preprocessing.py"

Step 5: Exporting the python path to add the object detection from tensorflow object detection API: 

    "export PYTHONPATH=$PYTHONPATH:~/repo/object_detection/object_detection/models/research/:~/repo/object_detection/object_detection/models/research/slim/"

Step 6: Run the following command in terminal from the directory "object_detection/models/research/":

    "protoc object_detection/protos/*.proto --python_out=."

Step 7: Run the script to check if everything is OK

    "python3 object_detection/builders/model_builder_test.py"

Step 8: Generate TF Records using "GeneratingTFRecords.py"

Step 9: Select and download the model using the "SelectingAndDownloadingModel.py"

Step 10: Configure Model Training Pipeline using "ConfigureModelTrainingPipeline.py"

Step 11: To launch tensorboard, execute the following command:

    "tensorboard --logdir=./models/training"

Step 12: Configuring the "pipeline_1.config" to copy from the existing sample of config file "/object_detection/models/research/object_detection/samples/config/ssd_mobilenet_v2_coco.config"

Step 13: Start the training running the script and giving the arguments:
    "python model_main.py --pipeline_config_path=./models/pretrained_model/pipeline_1.config --model_dir=./models/training"

Step 14: Change the classes and all the class id number in the corresponding files.

Step 15: Export the best model using the script "ExportTrainedModel.py".

Step 16: Use the frozen path from "models/fine_tuned_model" with the file name "frozen_inference_graph.pb" and label_map.pbtxt and run the UseModel.py

Change classes when training new model in:

1. pipeline.config
2. GeneratingTFRecords.py

Debugging:

Using TFRecord viewer for verifying the .record file, execute:

    "python3 tfviewer.py /home/mahimana/Documents/Deep_Learning/ObjectDetection/data/train_labels.record --labels-to-highlight='cubesat, rock, processing_plant'"


#Problems Faced: 
1. Error Generating TF Records, each bounding box was covering the whole image, "tf.io.TFRecordWriter" was causing problem, using "tf.python_io.TFRecordWriter" instead.

2. There was an error: "InvalidArgumentError: indices[0] = 0 is not in [0, 0)", this was due to the incompatible model with the object detection api. Then used other model to train on the dataset.

Step 17: To export tensorRT model, run the script:
    " python ConvertToTensorRTModel.py "

    Problems faced: 
    1. There was no attribute "score_threshold" in "pipeline.config" file in model.faster_rcnn.second_stage_post_processing, added "batch_non_max_suppression", in which there was an attribute called "score_threshold".