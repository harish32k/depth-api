FROM python:latest

ADD . /home/model-server/

WORKDIR /home/model-server/

RUN apt-get update && apt-get install -y python3-opencv

RUN pip3 install opencv-python

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -r requirements.txt

RUN python3 load_model.py

CMD exec gunicorn --bind 0.0.0.0:5000 --timeout=150 app:app -w 1

EXPOSE 5000