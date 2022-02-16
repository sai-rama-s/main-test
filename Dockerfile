    FROM python:3.7-slim-buster

    RUN apt-get update

    RUN apt-get install libopenblas-dev -y

    RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

    RUN apt-get -y install curl
    
    RUN apt-get install libgomp1    

    WORKDIR /app

    COPY requirements.txt .

    RUN pip3 install -r requirements.txt

    COPY . .

    CMD [ "python3", "main.py" ]