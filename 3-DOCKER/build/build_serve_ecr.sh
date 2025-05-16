#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
model_version='test' # Specify your model version
mode_name="container_docker"  # Specify your model name


if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi


# Check if the image already exists
if docker image inspect ${image} > /dev/null 2>&1; then
    echo "Docker image '${image}' already exists. Deleting the old image."
    docker rmi ${image}
fi


# Check if opt_program/pipeline exists, if not, copy ../pipeline to it
# if [ ! -d "opt_program/pipeline" ]; then
#     echo "'opt_program/pipeline' does not exist. Copying from '../pipeline'."
#     cp -r ../pipeline opt_program/pipeline
# fi


# Make the 'train' and 'serve' scripts executable, if they are part of your Docker container's build process.
# chmod +x opt_program/train
chmod +x opt_program/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-east-1 if none defined)
# set your region to be us-east-2
# aws configure set region us-east-2
region=$(aws configure get region)
region=${region:-us-east-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"


# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build --platform linux/amd64 \
             --build-arg MODEL_VERSION_ARG=${model_version} \
             --build-arg MODE_ARG=${mode_name} \
             -t ${image} .

echo "Docker image '${image}' built successfully."

docker tag ${image} ${fullname}

docker push ${fullname}
