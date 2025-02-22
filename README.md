# Spark Distributed Training Project

This project demonstrates how to set up a Spark cluster with multiple nodes and perform distributed training using TensorFlow, Horovod, and TensorFlowOnSpark.

## Project Structure

- `Dockerfile`: Defines the Docker image for the Spark master and worker nodes, including the installation of necessary dependencies.
- `docker-compose.yml`: Defines the services for the Spark master and worker nodes.
- `distribute_training.py`: The main script for distributed training using TensorFlow, Horovod, and TensorFlowOnSpark.
- `one_node.py`: A script for training on a single node.
- `txt`: Configuration for launching a Spark cluster with a single node.

## Prerequisites

- Docker
- Docker Compose

## Setup

1. **Build the Docker image**:

   ```sh
   docker-compose build

Start the Docker Compose services:

Verify the Spark Cluster:

Open your web browser and go to http://localhost:8080 to access the Spark UI. Ensure that all worker nodes are listed and have sufficient resources.

Running the Training Script
Access the Spark master container:

Submit the Spark job:

Single Node Training
If you want to run the training on a single node, use the one_node.py script and the configuration in txt.

Update the docker-compose.yml file to use a single node configuration:

Recreate the container:

Submit the single node training job:

Monitoring and Results
Spark UI: Access the Spark UI at http://localhost:8080 to monitor the job status and resource usage.
Model and Plots: The trained model and training plots will be saved in the /opt/models directory inside the container. You can mount this directory to your host machine using Docker volumes.
Troubleshooting
Port Conflicts: If you encounter port conflicts, update the port mappings in the docker-compose.yml file.
Resource Allocation: Ensure that the worker nodes have sufficient resources (CPU, memory) to run the job.
