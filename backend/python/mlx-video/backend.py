#!/usr/bin/env python3
"""
MLX Video Backend for LocalAI
Video generation using diffusers with MPS support for macOS
"""

import os
import sys
import tempfile
import logging
import traceback
import argparse
import json
import threading
from pathlib import Path

import grpc
from concurrent import futures

# Import local grpc generated files
import backend.proto.video_pb2 as video_pb2
import backend.proto.video_pb2_grpc as video_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLXVideoBackend(video_pb2_grpc.VideoServiceServicer):
    """MLX Video Generation Backend using diffusers with MPS support"""
    
    def __init__(self):
        self.model = None
        self.model_id = None
        self.device = None
        self.dtype = None
        self.lock = threading.Lock()
        
    def LoadModel(self, request, context):
        """Load the video generation model"""
        try:
            model_id = request.model_id
            logger.info(f"Loading MLX video model: {model_id}")
            
            with self.lock:
                if self.model is not None:
                    logger.warning(f"Model already loaded: {self.model_id}")
                    return video_pb2.LoadResponse(success=False, message="Model already loaded")
                
                # Import diffusers with MPS support
                try:
                    import torch
                    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
                    
                    # Determine device - prefer MPS on macOS
                    if torch.backends.mps.is_available():
                        self.device = "mps"
                        logger.info("Using MPS (Metal Performance Shaders) for acceleration")
                    elif torch.cuda.is_available():
                        self.device = "cuda"
                        logger.info("Using CUDA for acceleration")
                    else:
                        self.device = "cpu"
                        logger.warning("No GPU available, using CPU (slow)")
                    
                    # Set dtype based on device
                    if self.device == "mps":
                        self.dtype = torch.float16
                    elif self.device == "cuda":
                        self.dtype = torch.float16
                    else:
                        self.dtype = torch.float32
                    
                    logger.info(f"Loading pipeline from: {model_id}")
                    logger.info(f"Device: {self.device}, dtype: {self.dtype}")
                    
                    # Load the pipeline
                    self.model = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                        safety_checker=None,
                    )
                    
                    # Use DPMSolver for faster generation
                    self.model.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.model.scheduler.config
                    )
                    
                    # Move to device
                    self.model = self.model.to(self.device)
                    self.model_id = model_id
                    
                    logger.info(f"Model loaded successfully: {model_id}")
                    return video_pb2.LoadResponse(success=True, message=f"Model {model_id} loaded on {self.device}")
                    
                except ImportError as e:
                    logger.error(f"Import error: {e}")
                    return video_pb2.LoadResponse(success=False, message=f"Import error: {e}")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    traceback.print_exc()
                    return video_pb2.LoadResponse(success=False, message=f"Failed to load model: {e}")
                    
        except Exception as e:
            logger.error(f"Error in LoadModel: {e}")
            traceback.print_exc()
            return video_pb2.LoadResponse(success=False, message=str(e))
    
    def GenerateVideo(self, request, context):
        """Generate video from text prompt"""
        try:
            if self.model is None:
                return video_pb2.GenerateResponse(
                    success=False, 
                    message="Model not loaded. Call LoadModel first."
                )
            
            prompt = request.prompt
            negative_prompt = request.negative_prompt or ""
            num_frames = request.num_frames or 24
            height = request.height or 512
            width = request.width or 512
            num_inference_steps = request.num_inference_steps or 25
            guidance_scale = request.guidance_scale or 7.5
            seed = request.seed or 42
            
            logger.info(f"Generating video with prompt: {prompt[:50]}...")
            logger.info(f"Parameters: frames={num_frames}, size={width}x{height}, steps={num_inference_steps}")
            
            try:
                import torch
                from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
                from diffusers.utils import export_to_gif
                
                # For video generation, we might want to use AnimateDiff or similar
                # For now, generate frames and export as GIF
                
                # Set random seed
                generator = torch.Generator(device=self.device)
                generator = generator.manual_seed(seed)
                
                # Generate frames
                frames = []
                for i in range(num_frames):
                    frame_prompt = prompt
                    if num_frames > 1:
                        # Add slight variation for animation
                        frame_prompt = f"{prompt}, frame {i+1} of {num_frames}"
                    
                    with torch.inference_mode():
                        image = self.model(
                            prompt=frame_prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                    frames.append(image)
                
                # Export to GIF
                with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
                    output_path = tmp_file.name
                
                export_to_gif(frames, output_path)
                logger.info(f"Video saved to: {output_path}")
                
                # Read the file
                with open(output_path, 'rb') as f:
                    video_data = f.read()
                
                # Clean up
                os.unlink(output_path)
                
                return video_pb2.GenerateResponse(
                    success=True,
                    message="Video generated successfully",
                    video_data=video_data,
                    output_path=output_path
                )
                
            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                traceback.print_exc()
                return video_pb2.GenerateResponse(
                    success=False,
                    message=f"Video generation failed: {e}"
                )
                
        except Exception as e:
            logger.error(f"Error in GenerateVideo: {e}")
            traceback.print_exc()
            return video_pb2.GenerateResponse(success=False, message=str(e))


def serve(port):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    video_pb2_grpc.add_VideoServiceServicer_to_server(MLXVideoBackend(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"MLX Video Backend started on port {port}")
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description='MLX Video Backend for LocalAI')
    parser.add_argument('--port', type=int, default=50052, help='gRPC port')
    args = parser.parse_args()
    
    serve(args.port)


if __name__ == '__main__':
    main()
