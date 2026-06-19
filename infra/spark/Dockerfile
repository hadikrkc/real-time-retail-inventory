FROM eclipse-temurin:17-jdk-jammy

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pyspark==3.5.1 \
    psycopg2-binary==2.9.10

ENV PYSPARK_PYTHON=python3
ENV JAVA_HOME=/opt/java/openjdk
