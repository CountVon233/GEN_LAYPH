FROM ubuntu:22.04
RUN apt -y update && \
    apt -y install \
      build-essential \
      clang-format \
      clang-tidy \
      git \
      make \
      cmake \
      curl \
      zip \
      tar \
      perl \
      pkg-config \
      autoconf \
      doxygen \
      clangd \
      libomp-dev
