FROM eclipse-temurin:17-jdk-jammy

RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pyspark==3.5.1 \
    psycopg2-binary==2.9.10 \
    scikit-learn==1.5.2 \
    statsmodels==0.14.4 \
    numpy==1.26.4

# Pre-download Spark-Kafka connector JARs so spark-submit works offline
RUN mkdir -p /opt/spark-kafka-jars && \
    wget -q --tries=3 --waitretry=5 -P /opt/spark-kafka-jars \
        "https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar" \
        "https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar" \
        "https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar" \
        "https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar"

ENV PYSPARK_PYTHON=python3
ENV JAVA_HOME=/opt/java/openjdk
