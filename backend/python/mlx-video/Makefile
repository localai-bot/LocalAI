# MLX Video Backend Makefile
# Build and run targets for the MLX video generation backend

PYTHON ?= python3
PIP ?= $(PYTHON) -m pi
GRPC_TOOLS_VERSION ?= 1.21.9
GRPCIO_VERSION ?= 1.60.0

.PHONY: all build install test clean proto

all: build

build:
	@echo "Building MLX Video Backend..."
	@$(PYTHON) setup.py build

install:
	@echo "Installing MLX Video Backend..."
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .

test:
	@echo "Running tests..."
	@./test.sh

clean:
	@echo "Cleaning up..."
	@rm -rf build/ dist/ *.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

proto:
	@echo "Generating gRPC code..."
	@$(PYTHON) -m grpc_tools.protoc -I../../proto \
		--python_out=. --grpc_python_out=. \
		../../proto/video_pb2.proto
