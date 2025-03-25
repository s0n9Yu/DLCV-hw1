FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-devel

RUN mkdir /app

WORKDIR /app

COPY requirement.txt /app

RUN pip3 install -r requirement.txt