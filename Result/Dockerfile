#https://runnable.com/docker/python/dockerize-your-python-application

From python:3

ADD main.py /
ADD network.py /

RUN pip install numpy
RUN pip install matplotlib
#RUN pip install math
#RUN pip install timeit.default_timer

CMD [ "python", "./main.py" ]