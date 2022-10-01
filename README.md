
# Submission for ACM/IEEE TinyML Contest at ICCAD 2022

## Description

This repository contains a tensorflow implementation of a depthwise-separable model to detect two types of life-threatening ventricular arrhythmias (VAs):  Ventricular fibrillation (VF) and ventricular tachycardia (VT). More details on the competition and contest problem can be found [here](https://tinymlcontest.github.io/TinyML-Design-Contest/Problems.html)

# Model Architecture
The first choice we make in our model architecture is to use Depthwise separable convolutions (DSCNN) owing to their smaller memory-footprint and redued MAC operations. We use 5-layer DSCNN described in `models/keras_model.py`. In addition, to reduce the FLOPS consumed, we use the early-exit technique by [Ghanathe et al.](https://arxiv.org/abs/2207.06613)[1]. However, our preliminary analyses showed that given the problem size, the accuracy of the early and final exit are very close. Therefore, the early-exit model, with fewer layers and parameters suffices for the given problem. To emphasize the early-exit, we add a auxillary loss function at the early-exit. This extra loss, optimizes the model more towards the early-exit than the final exit. Moreover, the existence of the final exit imporves the robustness of the early-exit. 


This code uses five main scripts, described below, to train and test your model for the given dataset.

## How do I run these scripts?

##### Step1:Install the necessary tools. It is encouraged to use a virtual environment like *conda*.
Note: **Tensorflow 2.3** requires **python==3.7**

    conda create -n myenv python=3.7
    conda activate myenv
    pip install -r requirements.txt

##### Step2: Training
Prepare the training data

    python prepare_dataset.py --path_data=<path to dataset> --path_indices=<data indices dir>

Run train and test script
    
    python training_tf.py --model_save_name=<model_name>
This script loads a pretrained model and runs the training routine. Finally, the trained model is saved in `saved_models/`.

Test the model with the following script.
    
    python test_tf.py --model_save_name=saved_models/trained_dscnn_ref
    

**Note**: Due the non-determinism induced by the GPU, tensorflow and data generator during training the model accuracy varies on consecutive runs. `saved_modes/trained_dscnn_ref` is the best-performing model.

where `models` is a folder of model structure file, `saved_models` is a folder for saving your models, `data_indices` is a folder of data indices (the given training dataset has been partitioned into training and testing dataset, you can create more partitions of the training data locally for debugging and cross-validation), and `records` is a folder for saving the statistics outputs. The [TinyML Contest 2022 web-page](https://tinymlcontest.github.io/TinyML-Design-Contest/Problems.html) provides a description of the data files.


## Model Deployment and Validation
This model is tested on NUCLEO-L432KC board with STM32CubeMX and the package X-Cube-AI.
##### Step1: Convert the model to tflite

    python convert_to_tflite.py

The tflite model is created and saved in `saved_models/trained_dscnn_float.tflite`

Once we obtain the tflite model file, we could deploy the model on the board by following the instructions described in [README-Cube.md](link) and instructions described in  [How to validate X-CUBE-AI model on board.md](link). W
The other two metrics, **Flash occupation** and **Latency** could be obtained based on the reports from STM32CubeMX. 

**Note**: Do not forget to replace the C source code files with the files provided in `framework_x-cube-ai` before deployment. Also, while adding the network to the STM32CubeMX project, select *tflite* and browse to `saved_models/trained_dscnn_float.tflite` to add the network.


##### Step2: Run Validation
Once the model is loaded on the device, run the following script
    
    python validation.py --path_data=<path to dataset> --com=<COM PORT>
    
**NOTE**: If you specify the *path_data* argument, do not forget to include a `/` (or a `\` on windows) at the end of the path specified. Also, check the COM port on windows device manager and specify before running.



## How do I obtain the scoring?
After training your model and obtaining test outputs with above commands, you could evaluate the scores of your models using the scoring function specified in [TinyML Contest 2022 evaluation](https://tinymlcontest.github.io/TinyML-Design-Contest/Problems.html). 

## References
    [1] Ghanathe, Nikhil P., and Steve Wilton. "T-RECX: Tiny-Resource Efficient Convolutional Neural Networks with Early-Exit." arXiv preprint arXiv:2207.06613 (2022).