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
    rm -rf /var/lib/apt/lists/*

ENV ANACONDA_ROOT /opt/anaconda
ENV PATH $ANACONDA_ROOT/bin:$PATH

RUN curl https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -o ~/anaconda.sh && \
    bash ~/anaconda.sh -b -p $ANACONDA_ROOT && \
    rm ~/anaconda.sh

RUN /bin/bash -c "conda create -n py35 python=3.5 && \ 
    source activate py35 && \
    conda install -c https://conda.anaconda.org/kne pybox2d && \
    pip --no-cache-dir install \
        tensorflow \
        gym[all] \
        baselines"

COPY xorg.conf /usr/share/X11/xorg.conf.d/xorg.conf

WORKDIR /root

# Make py35 default environment
ENV DEFAULT_ANACONDA_ENV $ANACONDA_ROOT/envs/py35
ENV PATH $DEFAULT_ANACONDA_ENV/bin:$PATH

ENTRYPOINT [ "/bin/bash","-c" ]
CMD ["bash"]
