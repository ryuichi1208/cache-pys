FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -e .

# Default cache settings (can be overridden at runtime)
ENV CACHE_DEFAULT_TTL=3600
ENV CACHE_CLEANUP_INTERVAL=60
ENV CACHE_MAX_SIZE=100
ENV CACHE_STRATEGY=hybrid
ENV CACHE_WEIGHT_RECENCY=0.5
ENV CACHE_WEIGHT_FREQUENCY=0.5

CMD ["python", "sample.py"]
