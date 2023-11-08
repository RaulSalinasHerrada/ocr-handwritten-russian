FROM python:3.10-slim

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update

# packages for working on tesseract
RUN apt-get -y install ffmpeg libsm6 libxext6

# packages for working on textract
RUN apt-get -y install antiword
RUN apt-get -y install build-essential libpoppler-cpp-dev pkg-config
RUN apt-get -y install poppler-utils 

# install tesseract models (eng-est-rus)
RUN apt-get -y install tesseract-ocr-eng
RUN apt-get -y install tesseract-ocr-est
RUN apt-get -y install tesseract-ocr-rus

# #For reading EPS files
# RUN apt-get -y install ghostscript

COPY ./requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir --upgrade  -r /tmp/requirements.txt

WORKDIR /app
ADD . /app

CMD ["python3" ,"main.py"]