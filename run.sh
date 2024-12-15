#!/bin/bash

# Function to run tests
run_tests() {
    echo "Running tests..."
    python scripts/test_gaussian.py
}

# Function to run reconstruction
run_reconstruction() {
    echo "Running Gaussian Splatting reconstruction..."
    python examples/gaussian_reconstruction.py "$@"
}

# Check command
case "$1" in
    "test")
        run_tests
        ;;
    "reconstruct")
        shift  # Remove 'reconstruct' from arguments
        run_reconstruction "$@"
        ;;
    *)
        echo "Usage: $0 {test|reconstruct} [options]"
        echo "Options for reconstruct:"
        echo "  --input-file PATH      Input point cloud file"
        echo "  --output-dir DIR       Output directory"
        echo "  --num-gaussians N      Number of Gaussians"
        echo "  --iterations N         Number of optimization iterations"
        echo "  --learning-rate F      Learning rate"
        exit 1
        ;;
esac