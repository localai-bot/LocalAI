#!/usr/bin/env python3
"""
Test client for MLX Video Backend
"""

import grpc
import sys
import os

# Add parent directory to path for proto imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import backend.proto.video_pb2 as video_pb2
import backend.proto.video_pb2_grpc as video_pb2_grpc


def test_load_model():
    """Test loading a model"""
    channel = grpc.insecure_channel('localhost:50052')
    stub = video_pb2_grpc.VideoServiceStub(channel)
    
    request = video_pb2.LoadRequest(
        model_id="runwayml/stable-diffusion-v1-5"
    )
    
    response = stub.LoadModel(request)
    print(f"LoadModel response: success={response.success}, message={response.message}")
    return response.success


def test_generate_video():
    """Test generating a video"""
    channel = grpc.insecure_channel('localhost:50052')
    stub = video_pb2_grpc.VideoServiceStub(channel)
    
    request = video_pb2.GenerateRequest(
        prompt="A beautiful sunset over the mountains",
        negative_prompt="blurry, low quality",
        num_frames=8,
        height=512,
        width=512,
        num_inference_steps=10,
        guidance_scale=7.5,
        seed=42
    )
    
    response = stub.GenerateVideo(request)
    print(f"GenerateVideo response: success={response.success}, message={response.message}")
    
    if response.success and response.video_data:
        # Save the generated video
        with open("test_output.gif", "wb") as f:
            f.write(response.video_data)
        print("Video saved to test_output.gif")
    
    return response.success


def main():
    print("Testing MLX Video Backend...")
    
    # Test 1: Load model
    print("\n--- Test 1: Load Model ---")
    load_success = test_load_model()
    if not load_success:
        print("WARNING: Model loading test failed (expected if backend not running)")
        return
    
    # Test 2: Generate video
    print("\n--- Test 2: Generate Video ---")
    generate_success = test_generate_video()
    
    if generate_success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Video generation test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
