# Use the open-source verification base image which includes Verilator, Icarus, etc.
FROM ghcr.io/hdl/sim/osvb

WORKDIR /app

# Install Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-requests \
    && rm -rf /var/lib/apt/lists/*

# Install requests library for API calls
RUN pip3 install --no-cache-dir requests

# Create /code directory for mounted volumes
RUN mkdir -p /code && chmod 777 /code

# Copy the agent code
COPY agent.py /app/agent.py

# Make it executable
RUN chmod +x /app/agent.py

# Set environment variables (can be overridden at runtime)
ENV SLM_API_URL=http://host.docker.internal:8000
ENV SLM_MODEL=phi35
ENV SLM_MAX_LENGTH=8192
ENV SLM_TIMEOUT=300

# Set the entrypoint
ENTRYPOINT ["python3", "/app/agent.py"]
