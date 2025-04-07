# THIS CONTAINER IS A NPM, RUST, PYTHON, DOCKER INTO ONE CONTAINER
# THIS GENERAL CONTAINER IS THE CORE OF val, USE IT AS YOU WISH AT YOUR OWN RISK

FROM ubuntu:22.04

# SYSTEM ENVIRONMENT
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# INSTALL DOCKER
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
RUN groupadd docker || true
RUN usermod -aG docker root
EXPOSE 2375

# MODULE ENVIRONMENT
WORKDIR /app
COPY . .
RUN pip install -e ./

# ENTRYPOINT (default to container running)
ENTRYPOINT ["tail", "-f", "/dev/null"]