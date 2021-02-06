FROM tensorflow/tensorflow:1.7.1-gpu-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq git sudo wget

RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip install jupyter_contrib_nbextensions==0.5.0 && jupyter contrib nbextension install --system && jupyter nbextension enable --system spellchecker/main

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# RUN git clone https://github.com/Calysto/notebook-extensions.git /tmp/notebook-extensions && cd /tmp/notebook-extensions && jupyter nbextension install --system calysto && jupyter nbextension enable --system calysto/spell-check/main

RUN mkdir /code
ENV HOME=/code

WORKDIR /code
