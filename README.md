# Project Inrtroduction

In this project, I will demonstrate how to fine-tune an image classification model using ResNet-34, a widely-used deep learning architecture for image-related tasks. ResNet-34 is known for its effectiveness in handling complex image data through its residual learning framework, making it a popular choice for image classification tasks. The dataset I will use for this demonstration is CIFAR-100, which consists of 100 classes of images, providing a challenging and diverse set of samples for training and evaluation.

# Project Setup Instructions

This project will use Jupyter Notebook for running the code, and the training and inference of the model will be performed on the SageMaker platform. You can upload the Jupyter notebook file to a SageMaker JupyterLab or Notebook instance, or you can run it on your local machine or EC2 instance ( In that case, you may not be able to use the sagemaker.get_execution_role() function. Instead, you should hard-code the ARN of the role you create.)

Regardless of where you run the code, you will need to prepare an IAM role with permissions for both Amazon S3 and SageMaker.

# Project Structure

This project contains four code files:

| file name | description |
| ---------- | ---------- |
| train_and_deploy.ipynb | The Jupyter notebook used to prepare the data, train the model, perform hyperparameter tuning, and deploy the model. |
| hpo.py | The Python script used for hyperparameter optimization (HPO). |
| train_model.py | The Python script used to train the model. This file is similar to hpo.py, with the primary difference being that it includes additional code for SageMaker Debugger and Profiler, which are not present in hpo.py|
| inference.py | The entry point for the SageMaker endpoint. Since we deploy a custom-trained model, this script helps process the input data, execute predictions, and send the response|

# Model Insights

This model demonstrates a fine-tuned version of ResNet-34 for image classification using the CIFAR-100 dataset. The training and evaluation process is monitored using SageMaker Debugger, which captures metrics such as batch loss, batch accuracy, epoch loss, and epoch accuracy. 
And the insights are as follow: 
| Name & Age     | City         |
|----------------|--------------|
| Alice, 25      | New York     |
|                | Los Angeles  |


 
