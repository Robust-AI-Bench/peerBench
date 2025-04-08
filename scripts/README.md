 # start of file
# Val Scripts

This directory contains utility scripts for managing, testing, and deploying the Val framework.

## Docker Scripts

Located in the `docker/` subdirectory, these scripts help manage Docker containers for Val:

- `build.sh`: Builds the Docker image for Val
- `start.sh`: Starts the Val container
- `stop.sh`: Stops the running Val container
- `enter.sh`: Opens a shell inside the running Val container
- `test.sh`: Runs tests inside the Val container
- `install.sh`: Installs dependencies inside the container

## Usage

These scripts are designed to be called through the Makefile in the root directory:

```bash
# Build the Docker image
make build

# Start the container
make start

# Stop the container
make stop

# Enter the running container
make enter

# Run tests inside the container
make test
```

## Adding New Scripts

When adding new scripts:

1. Place them in the appropriate subdirectory
2. Make them executable with `chmod +x script_name.sh`
3. Update the Makefile if needed to include the new script

You can use the `make chmod` command to make all scripts executable at once.
