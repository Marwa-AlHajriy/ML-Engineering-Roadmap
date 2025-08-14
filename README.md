# ML-Engineering-Roadmap


A personalized roadmap from statistician to ML engineer (fill in software engineering gaps and upskill in AI)

---

## Stage 1: Python OOP (refresher)
- Python OOP Tutorials - Working with Classes by Corey Schafer (https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc) 

---

## Stage 2: Model Deployment + REST APIs
- Deployment of ML Models by Krish Naik (https://www.youtube.com/playlist?list=PLZoTAELRMXVOAvUbePX1lTdxQR8EY35Z1)

When deploying machine learning models, it's important to understand the environment where the model will run. Common deployment environments:

**On-Premises Deployment**
- **Definition**: Model is deployed on the organization's own physical servers.
- **Use Case**: Organizations with strict data security requirements (e.g., banks, hospitals).
- **Pros**: Full control over data, no third-party dependency.
- **Cons**: Expensive infrastructure, limited scalability, high maintenance.

**Infrastructure as a Service (IaaS)**
- **Definition**: Cloud-based virtual machines where you manage the OS and environment (e.g., AWS EC2, Google Compute Engine).
- **Use Case**: Teams wanting more control without owning physical servers.
- **Pros**: Scalable, flexible, full environment control.
- **Cons**: Still responsible for updates, monitoring, and configurations.

**Platform as a Service (PaaS)**
- **Definition**: Managed platforms where you only provide your code or model (e.g., Heroku, AWS Elastic Beanstalk, Google App Engine).
- **Use Case**: Fast deployment with minimal infrastructure concerns.
- **Pros**: Easy to use, auto-scaling, minimal setup.
- **Cons**: Less customizable, higher cost for large-scale apps.



---


## Stage 3: Create end-to-end ML project on cloud (AWS)

- Tutorial guide: End To End Machine Learning Project Implementation Using AWS Sagemaker by Krish Naik (https://www.youtube.com/watch?v=Le-A72NjaWs)
- We will work on a different model, a classification CNN model using Brain Tumor MRI Dataset form Kaggle (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download)

  1. Create Iam user (Administrative access)
     - Given Access key and Secret Access key to securely connect terminal with aw (next step)
  
  2. AWS CLI configure (a command line interfaceto interact with AWS using commands in terminal)
     - In terminal:
       ```bash
       aws configure
       ```
     - Will be asked to enter Access key, Secret Access key, Default region name & Default output format
     - To check current config:
       ```bash
       cat ~/.aws/config
       ```
  3. Set up local project environment (VS Code)
   - On Desktop, created a new project folder/directory, this is where all project files will be stores (data, requirements, gitignore, etc.):
     ```bash
     mkdir CNN_Brain_Tumor
     ```
   - Opened the `CNN_Brain_Tumor` folder in VSCode
   - In VSCode terminal, created a virtual environment inside the project folder:
     ```bash
     python3 -m venv mycnnexenv
     ```
   - Activated the environment:
     ```bash
     source mycnnexenv/bin/activate  
     ```
   - Created a `requirements.txt` file with the necessary packages:
   - Installed dependencies:
     ```bash
     pip install -r requirements.txt
     ```
4. Create Jupyter notebook inside VS Code

   - In the same `CNN_Brain_Tumor` folder, created a notebook file:
     ```
     cnn_brain_tumor.ipynb
     ```
   - Clicked "Select Kernel" and selected the virtual environment (`mycnnexenv`)
   - Created an S3 bucket named `cnnmodelbucketforsagemaker`.
       - **Amazon S3** is used to store datasets, model artifacts, and logs.
       - **Amazon SageMaker** will access data from this bucket to train and deploy the CNN model in the cloud.
   - Started coding the ML pipeline using:
     ```python
     import sagemaker
     import boto3   # will be used to connect the bucket
     import tensorflow as tf
     import matplotlib.pyplot as plt
     import numpy as np
     import pandas as pd
     sm_boto3 = boto3.client("sagemaker")
     sess = sagemaker.Session()
     region = sess.boto_session.region_name
     bucket = 'cnnmodelbucketforsagemaker'
     print('Using bucket:', bucket)
     ```

5. Uploaded dataset to S3 bucket
   - Prepared zip files of each set after splitting data into train, val, test
   - Upload zipped files to S3 bucket:
     ```python
     prefix = 'sagemaker/cnn_brain_tumor_project'
     s3_train_path = sess.upload_data('data/Train.zip', bucket=bucket, key_prefix=prefix)
     s3_val_path   = sess.upload_data('data/Val.zip',   bucket=bucket, key_prefix=prefix)
     s3_test_path  = sess.upload_data('data/Test.zip',  bucket=bucket, key_prefix=prefix)
  
     ```

6. Created Sagemaker Training script (script.py)
	-	Unzips train/val/test data inside the container.
	-	Builds TensorFlow datasets using image_dataset_from_directory.
	-	Defines and trains the CNN model with augmentation and normalization.
	-	Evaluates the model on the test set.
	-	Saves:
    	-	Model in SavedModel format for TensorFlow Serving
    	-	class_names.json for inference
    	-	learning_curves.png plot of training vs validation accuracy/loss
    	-	training_history.json with metrics
    	-	Confusion matrix and classification report 
       
