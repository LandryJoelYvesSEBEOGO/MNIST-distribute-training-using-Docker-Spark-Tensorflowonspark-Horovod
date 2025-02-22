# Base image for Spark
FROM openjdk:11-jre-slim

# Install wget and other dependencies
RUN apt-get update && apt-get install -y wget build-essential

# Install CMake
RUN apt-get install -y cmake

# Install Python3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install OpenMPI
RUN apt-get install -y openmpi-bin libopenmpi-dev

# Install TensorFlow
RUN pip install tensorflow==2.12.0

# Install Horovod with TensorFlow support
RUN HOROVOD_WITH_TENSORFLOW=1 pip3 install horovod

# Install TensorFlowOnSpark
RUN pip install tensorflowonspark

# Install additional Python frameworks
RUN pip install matplotlib seaborn numpy scikit-learn

# Define environment variables
ENV SPARK_VERSION=3.4.0
ENV HADOOP_VERSION=3.2

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Download and install Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz && \
    tar -xzf spark-3.4.0-bin-hadoop3.tgz -C /opt && \
    mv /opt/spark-3.4.0-bin-hadoop3 $SPARK_HOME && \
    rm spark-3.4.0-bin-hadoop3.tgz

# Expose Spark ports
EXPOSE 4040 7077 8080 18080

# Set the entrypoint for the container
CMD ["/opt/spark/bin/spark-class", "org.apache.spark.deploy.master.Master"]