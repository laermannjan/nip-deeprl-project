# Dockerfile to setup OpenAI Gym, Baselines and Tensorflow on Ubuntu
FROM ubuntu:14.04

LABEL maintainer "Jan Laermann [laermannjan@gmail.com]"

RUN apt-get update && apt-get install -y \
        cmake \
        curl \
        git \
        libav-tools \
        libboost-all-dev \
        libjpeg-dev \
        libpq-dev \
        libsdl2-dev \
        python-numpy \
        python-opengl \
        python-scipy \
        python-pyglet \
        python-setuptools \
        swig \
        unzip \
        wget \
        xorg-dev \
        xpra \
        xvfb \
        zlib1g-dev

RUN echo deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse \
    | sudo tee /etc/apt/sources.list.d/box2d-py-swig.list && \
    apt-get update && \
    apt-get install -t trusty-backports swig3.0  &&\
    apt-get remove -y swig swig2.0 && \
    ln -s /usr/bin/swig3.0 /usr/bin/swig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

ENV MINICONDA_ROOT /opt/miniconda
ENV PATH $MINICONDA_ROOT/bin:$PATH

RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/anaconda.sh && \
    bash ~/anaconda.sh -b -p $MINICONDA_ROOT && \
    rm ~/anaconda.sh

RUN /bin/bash -c "conda create -n py35 python=3.5 && \ 
    source activate py35 && \
    conda install -c https://conda.anaconda.org/kne pybox2d && \
    pip --no-cache-dir install \
        tensorflow \
        gym[classic_control] \
        gym[box2d] \
        baselines"


WORKDIR /code
ENV PROJECT_PATH /code

COPY . /code
COPY xorg.conf /usr/share/X11/xorg.conf.d/xorg.conf

RUN chmod +x /code/docker-entrypoint.sh


ENTRYPOINT ["/code/docker-entrypoint.sh"]
