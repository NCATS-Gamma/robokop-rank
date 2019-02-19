# This Dockerfile is used to build ROBOKOP

FROM python:3.6.6-stretch

LABEL maintainer="patrick@covar.com"
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV REFRESHED_AT 2018-05-10

## Install basic tools
RUN apt-get update
RUN apt-get install -yq \
    vim

## Set up home directory
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID murphy
RUN useradd -m -u $UID -g $GID -s /bin/bash murphy
WORKDIR /home/murphy

## Get ROBOKOP software
RUN git clone https://github.com/NCATS-Gamma/robokop-rank.git
WORKDIR /home/murphy/robokop-rank

## Install all requirements
RUN pip install -r ./requirements.txt --src /usr/local/src

## Finish up
ENV HOME=/home/murphy
ENV USER=murphy

WORKDIR /home/murphy
RUN rm -r /home/murphy/robokop-rank

ENTRYPOINT ["./robokop-rank/deploy/ranker/startup.sh"]
CMD ["supervisord", "-c", "./robokop-rank/deploy/ranker/supervisord.conf"]