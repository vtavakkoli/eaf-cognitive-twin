FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY configs /app/configs
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "eaf_twin.cli"]
CMD ["run", "--config", "configs/base_case.json", "--output-dir", "outputs"]
