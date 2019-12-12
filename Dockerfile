FROM ubuntu
MAINTAINER grenwi

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
RUN apt-get -y install python3-pip
RUN pip3 install pybind11
# RUN pip3 install git+https://github.com/facebookresearch/fastText.git
ADD requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://www.github.com/keras-team/keras-contrib.git
ADD scripts/*.py scripts/
ADD embeddings/wiki.de.bin embeddings/wiki.de.bin
COPY models/ models/
COPY templates/ templates/
ADD app.py .

ENV FLASK_APP app.py
ENV FLASK_DEBUG 0
# CMD flask run --host 0.0.0.0 --port 5001 --with-threads
CMD gunicorn -b 0.0.0.0:5001 --worker-connections 1000 --timeout 180 app:app
