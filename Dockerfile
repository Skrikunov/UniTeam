
FROM python:3.7.10

WORKDIR /usr/src/app

RUN apt-get update \
    && apt-get install -y python-opencv \
    && pip install numpy \
    && pip install opencv-python \
    && pip install mediapipe \
    && pip install imutils  

COPY UDPc.py ./

#ARG UDP_IP="192.168.223.24
#ARG UDP_PORT=5002


# only client
# !python UDPc.py -i {UDP_IP} -p {UDP_PORT} -f 1 -s 0
# CMD [ "python", "./UDPc.py", "-i" ,"192.168.223.24","-p","5002","-f" ,"1","-s","0"]
CMD [ "python", "./UDPc.py", "-i" ,"10.16.116.12","-p","5002","-f" ,"1","-s","0"]
