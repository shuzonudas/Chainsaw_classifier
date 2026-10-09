FROM python:3.13.14-trixie

ADD ML_app.py .
ADD ML_model ./ML_model
ADD requirements.txt .

# RUN pip install requests numpy pandas librosa tensorflow
RUN pip install -r requirements.txt

CMD [ "python","./ML_app.py" ]

