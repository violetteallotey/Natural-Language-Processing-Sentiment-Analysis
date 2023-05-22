FROM python:3.9

WORKDIR /app


RUN mkdir -p /.cache/huggingface/hub && chmod -R 777 /.cache

 
ENV TRANSFORMERS_CACHE /.cache/huggingface/hub

 
COPY requirements.txt .


COPY sentimentapp.py .

 
COPY senti.jpg .
COPY negative-smiley-face.png .
COPY positive-smiley-face.png .
COPY neutral-smiley-face.png .
COPY swipe-swoosh.mp3 .


RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt



CMD ["streamlit","run","sentimentapp.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--browser.serverAddress", "Adoley/app_personal.hf.space", "--browser.serverAddress","0.0.0.0:7860"]