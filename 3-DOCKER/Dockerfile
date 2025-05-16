# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

# Use the official Bitnami Python 3.7 image as the base
# This provides a stable and secure foundation for our Docker image
# FROM public.ecr.aws/bitnami/python:3.7
FROM python:3.11-slim

# Specify the maintainer of the Dockerfile
# This is useful for users who might have questions or need support
# MAINTAINER Amazon AI <sage-learner@amazon.com>

# Update the package lists and install necessary packages
# - wget: for downloading files
# - python3-pip: Python package installer
# - python3-setuptools: for handling Python packages
# - nginx: web server for serving the application
# - ca-certificates: for SSL/TLS certificates
# The --no-install-recommends flag minimizes the number of installed packages
# Cleaning up apt lists reduces the image size
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python and pip
# This allows us to use 'python' and 'pip' commands instead of 'python3' and 'pip3'
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python packages required for the application
# --no-cache-dir flag prevents caching of installation files, reducing image size
# Specific versions are pinned to ensure reproducibility
# Example: numpy==1.16.2 installs version 1.16.2 of numpy
# RUN pip --no-cache-dir install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask gunicorn



# Set environment variables
# PYTHONUNBUFFERED=TRUE ensures Python output is sent straight to terminal (helps with logging)
# PYTHONDONTWRITEBYTECODE=TRUE prevents Python from writing .pyc files (reduces image size)
# PATH is updated to include /opt/program, where our application code will reside

ARG MODEL_VERSION_ARG
ARG MODE_ARG

ENV MODEL_VERSION=${MODEL_VERSION_ARG:-"vTest"}
ENV MODE=${MODE_ARG:-"container-docker"}

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/program:${PATH}"

# Copy the application code into the Docker image
# This assumes there's a directory named 'linear_regression' in the same directory as the Dockerfile
# COPY linear_regression /opt/program
COPY opt_program /opt/program

# Set the working directory to where the application code is located
# This means subsequent commands will be run from this directory
WORKDIR /opt/program


RUN pip install -r requirements.txt \
    --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org 

RUN pip install torch torchvision torchaudio \
    --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org


