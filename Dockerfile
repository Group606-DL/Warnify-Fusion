# syntax=docker/dockerfile:1

#FROM python:3.8.5-alpine
FROM python:3.8-slim-buster
WORKDIR /app
ENV PYTHONUNBUFFERED=1


ENV FLASK_APP=Fusion.py
ENV FLASK_RUN_HOST=0.0.0.0

#RUN apt-get update
#RUN apt install -y libgl1-mesa-glx
#sudo apt-get install libglib2.0-0

RUN pip install -U opencv-python
RUN apt-get upgrade
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 5000
COPY . .

CMD ["python3", "Fusion.py"]
