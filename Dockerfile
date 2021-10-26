FROM python:3.6
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY req.txt .
RUN pip install -r req.txt
COPY app.py .
EXPOSE 80
ENV PORT 80
CMD ["streamlit", "run","app.py","--server.port","$PORT"]
