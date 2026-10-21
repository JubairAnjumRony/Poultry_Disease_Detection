# End-to-End-Deep-Learning-Project-Chicken-Disease


#### Data Link: [Donwload Link](https://drive.google.com/file/d/1pV0DAdyjzsjk0HL7f8_5qiS_mVyjYk25/view?usp=sharing)

## Workflows

1. Update config.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py




# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/entbappy/End-to-End-Deep-Learning-Project-Chicken-Disease
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### STEP 03- GPU setup (NVIDIA GTX 1650 / CUDA 11.8)

`tensorflow==2.12.0` needs CUDA 11.8 and cuDNN 8.6. On Linux these are installed
inside the virtualenv from `requirements.txt` (`nvidia-*-cu11` wheels), so no
system-wide CUDA toolkit is required. The libraries must be added to
`LD_LIBRARY_PATH` before Python starts; the activation script does this for you:

```bash
source scripts/activate_gpu.sh
```

The project also preloads these libraries automatically when `cnnClassifier` is
imported (`src/cnnClassifier/gpu.py`), so `.venv/bin/python app.py` and
`.venv/bin/python main.py` use the GPU even without sourcing the activation
script. Running through the activated venv remains the recommended workflow.

Verify the GPU is visible:

```bash
python gpu_test.py
```

The output should contain `PhysicalDevice(name='/physical_device:GPU:0', ...)`
and `Result device: .../GPU:0`. `python main.py` and `python app.py` will then
run on the GPU automatically.

> The GTX 1650 has 4 GB of VRAM. The default `BATCH_SIZE: 8` in `params.yaml`
> is a safe value for the VGG16 model at 224x224.


### Export the environment
```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID""
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"

```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```




# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

