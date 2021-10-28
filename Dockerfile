FROM ubuntu:18.04

ENV RUBY_VERSION 3.0.2
ENV DEBIAN_FRONTEND=noninteractive

# Using aliyun ubuntu mirror
RUN sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list 
RUN sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list 
RUN sed -i "s/security.debian.org/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list 

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        # imagemagick \
        libczmq-dev \
        libffi-dev \
        libreadline-dev \
        libsox-dev \
        libsox-fmt-all \
        libssl-dev \
        libtool \
        libvips \
        libyaml-dev \
        libzmq3-dev \
        make \
        python3 \
        python3-pip \
        python3-setuptools \
        sox \
        unzip \
        wget \
        zlib1g-dev \
        libjpeg-dev \
        vim \
        tcl-dev \
        tk-dev \
        python3-tk \
        && \
    rm -rf /var/lib/apt/lists/*
        

# install Ruby
RUN cd /tmp && \
    wget -O ruby.tar.gz -q https://cache.ruby-lang.org/pub/ruby/3.0/ruby-$RUBY_VERSION.tar.gz && \
    mkdir ruby && \
    tar xfz ruby.tar.gz -C ruby --strip-components=1 && \
    rm ruby.tar.gz && \
    cd ruby && \
    ./configure --disable-install-doc --enable-shared && \
    make -j && \
    make install && \
    cd .. && \
    rm -r ruby && \
    ruby --version && \
    bundle --version

# install LibTorch
RUN cd /opt && \
    wget -O libtorch.zip -q https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu102.zip && \
    unzip -q libtorch.zip && \
    rm libtorch.zip

# Using douban pipy mirror
RUN pip3 install -i https://pypi.douban.com/simple/ -U pip && \
    pip3 config set global.index-url https://pypi.douban.com/simple/ && \
    pip3 install matplotlib

# install gems
RUN gem install --verbose -v 0.8.1 torch-rb -- --with-torch-dir=/opt/libtorch && \
    gem install torchaudio -- --with-torch-dir=/opt/libtorch && \
    gem install torchtext torchvision && \
    gem install pycall && \
    gem install matplotlib























