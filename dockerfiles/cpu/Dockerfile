FROM ubuntu:bionic

MAINTAINER Leszek Pryszcz <lpryszcz@crg.es>

ARG PACKAGE_VERSION=1.2.0
ARG PYTHON_VERSION=3.7.6
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y build-essential zlib1g-dev curl liblzma-dev libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev
RUN curl https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz -o Python-${PYTHON_VERSION}.tgz
RUN tar -zxf Python-${PYTHON_VERSION}.tgz; rm Python-${PYTHON_VERSION}.tgz ; cd Python-${PYTHON_VERSION}; ./configure; make; make install
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python; ln -s /usr/local/bin/pip3 /usr/local/bin/pip

#Installing modules for Deeplexicon
RUN pip install --upgrade pip setuptools
RUN pip install h5py==2.10 Keras==2.2.4 Pandas PyTs==0.8.0 Scikit-learn numba==0.53 TensorFlow==1.13.1

#Installing Deeplexicon
RUN apt-get install -y git liblzma-dev
RUN git clone --depth 1 https://github.com/lpryszcz/deeplexicon.git #&& rm -r deeplexicon/.git
RUN chmod +x deeplexicon/deeplexicon*.py
ENV PATH "$PWD/deeplexicon/:${PATH}"

# Cleanup
RUN apt-get clean && apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* && rm -r ~/.cache/pip 
RUN locale-gen en_US.UTF-8
