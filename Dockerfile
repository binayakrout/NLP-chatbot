FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

COPY * /app

ADD templates /app

ADD source /app

ADD destination /app

EXPOSE 8080

CMD [ "python3", "/app/app.py"]
