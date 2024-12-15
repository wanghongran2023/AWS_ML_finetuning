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

| Item | Description |
| --------------- | ---------- |
| Model Design | ResNet-34 with an additional fully connected (fc) layer. Residual blocks 1 to 3 of the pretrained ResNet-34 are frozen. |
| Training Data | CIFAR-100 train dataset, which contains 50,000 images across 100 classes |
| Test Data | CIFAR-100 test dataset, which contains 10,000 images across 100 classes |
| Hyperparameters | For the demo, we tuned three hyperparameters: learning_rate, batch_size, and epochs 10 times. The best hyperparameter set found was {'learning_rate': 0.017483127170473314, 'batch_size': 128, 'epochs': 70} |
| Training Instance Type | ml.p3.2xlarge (1 NVIDIA Tesla V100 and 8 CPUs) |
| Final Validation Accuracy | 40.71% |
| Loss at Final Epoch | 1.5454 |
| Final Training Accuracy | 58.19% |


 
