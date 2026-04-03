FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY eaf_simulator.py /app/
ENTRYPOINT ["python", "eaf_simulator.py"]
CMD ["--output-dir", "outputs"]
