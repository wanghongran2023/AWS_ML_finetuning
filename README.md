# Project Inrtroduction

In this project, I will demonstrate how to fine-tune an image classification model using ResNet-34, a widely-used deep learning architecture for image-related tasks. ResNet-34 is known for its effectiveness in handling complex image data through its residual learning framework, making it a popular choice for image classification tasks. The dataset I will use for this demonstration is CIFAR-100, which consists of 100 classes of images, providing a challenging and diverse set of samples for training and evaluation.

# Project Setup Instructions

This project will use Jupyter Notebook for running the code, and the training and inference of the model will be performed on the SageMaker platform. You can upload the Jupyter notebook file to a SageMaker JupyterLab or Notebook instance, or you can run it on your local machine or EC2 instance ( In that case, you may not be able to use the sagemaker.get_execution_role() function. Instead, you should hard-code the ARN of the role you create.)

Regardless of where you run the code, you will need to prepare an IAM role with permissions for both Amazon S3 and SageMaker.
