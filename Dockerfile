FROM python:3.9-slim-bullseye

ADD credentials /credentials

WORKDIR /classify
ADD classify .
RUN pip install .
