# Use the official Python image
FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app/
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 80

COPY ./app /app/

CMD ["uvicorn", "server_d:app", "--host", "0.0.0.0", "--port", "80"]
