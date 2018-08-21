FROM ubuntu
MAINTAINER ragvri

RUN apt-get update 
RUN apt-get -y install software-properties-common
RUN apt-get update
RUN apt-get install python3.5
RUN apt-get clean && apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

WORKDIR /app/
EXPOSE 5001

RUN apt-get -y install git-core
RUN git clone https://github.com/facebookresearch/fastText.git 
RUN apt-get -y install python3-pip
RUN python3 -m pip install pybind11
RUN cd fastText && python3 setup.py install
RUN cd ..
RUN apt-get install wget
RUN wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.zip
RUN apt-get -y install unzip
RUN unzip wiki.de.zip
RUN rm -rf wiki.de.zip
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
ADD generator_NER_best.h5 .
RUN python3 -m pip install jupyter
RUN python3 -m pip install git+https://www.github.com/keras-team/keras-contrib.git


