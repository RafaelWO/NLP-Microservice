FROM pytorch/pytorch
WORKDIR /usr/src/app

COPY . .

RUN pip install -r ./requirements.txt
RUN python ./src/download.py

ENTRYPOINT [ "python" ]
CMD [ "src/service.py" ]
