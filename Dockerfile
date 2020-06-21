FROM python:3.8.0-alpine
USER root:root
WORKDIR /root

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
ENV PYTHONUNBUFFERED 0
ENV ARCHFLAGS -Wno-error=unused-command-line-argument-hard-error-in-future

COPY --chown=root:root requirements.txt .
RUN apk update \
	&& apk add --no-cache \
	curl\
	gcc\
	gfortran\
	g++\
	python3-dev\
	make\
	automake\
	&& pip install -U pip \
	&& pip install cython \
	&& pip install -r requirements.txt
