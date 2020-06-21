FROM python:3.8.0-alpine
USER root:root
WORKDIR /root

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
ENV PYTHONUNBUFFERED 0

COPY --chown=root:root requirements.txt .
RUN apk update \
	&& apk add --no-cache \
	curl \
	&& pip install -U pip \
	&& pip install -r requirements.txt
