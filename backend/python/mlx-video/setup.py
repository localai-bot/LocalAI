#!/usr/bin/env python3
"""
Setup script for MLX Video Backend
"""

from setuptools import setup, find_packages

setup(
    name="mlx-video-backend",
    version="0.1.0",
    description="MLX Video Backend for LocalAI - Video generation with MPS support",
    author="LocalAI Contributors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "grpcio>=1.60.0",
        "grpcio-tools>=1.60.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "torch>=2.1.0",
        "diffusers>=0.24.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "imageio>=2.33.0",
        "imageio-ffmpeg>=0.4.9",
    ],
    entry_points={
        "console_scripts": [
            "mlx-video-backend=backend:main",
        ],
    },
)
