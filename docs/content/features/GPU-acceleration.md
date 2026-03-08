+++
disableToc = false
title = "⚡ GPU acceleration"
weight = 9
url = "/features/gpu-acceleration/"
+++

This section contains instructions on how to use LocalAI with GPU acceleration across different hardware platforms.

## Automatic Backend Detection

When you install a model from the gallery (or a YAML file), LocalAI intelligently detects the required backend and your system's capabilities, then downloads the correct version for you. Whether you're running on a standard CPU, an NVIDIA GPU, an AMD GPU, or an Intel GPU, LocalAI handles it automatically.

For advanced use cases or to override auto-detection, you can use the `LOCALAI_FORCE_META_BACKEND_CAPABILITY` environment variable. Here are the available options:

- `default`: Forces CPU-only backend. This is the fallback if no specific hardware is detected.
- `nvidia`: Forces backends compiled with CUDA support for NVIDIA GPUs.
- `amd`: Forces backends compiled with ROCm support for AMD GPUs.
- `intel`: Forces backends compiled with SYCL/oneAPI support for Intel GPUs.

## Model configuration

Depending on the model architecture and backend used, there might be different ways to enable GPU acceleration. It is required to configure the model you intend to use with a YAML config file. For example, for `llama.cpp` workloads a configuration file might look like this (where `gpu_layers` is the number of layers to offload to the GPU):

```yaml
name: my-model-name
parameters:
  # Relative to the models path
  model: llama.cpp-model.ggmlv3.q5_K_M.bin

context_size: 1024
threads: 1

f16: true # enable with GPU acceleration
gpu_layers: 22 # GPU Layers (only used when built with cublas)

```

For diffusers instead, it might look like this instead:

```yaml
name: stablediffusion
parameters:
  model: toonyou_beta6.safetensors
backend: diffusers
step: 30
f16: true
diffusers:
  pipeline_type: StableDiffusionPipeline
  cuda: true
  enable_parameters: "negative_prompt,num_inference_steps,clip_skip"
  scheduler_type: "k_dpmpp_sde"
```

## CUDA (NVIDIA) acceleration

### Requirements

Requirement: nvidia-container-toolkit (installation instructions [1](https://www.server-world.info/en/note?os=Ubuntu_22.04&p=nvidia&f=2) [2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

If using a system with SELinux, ensure you have the policies installed, such as those [provided by nvidia](https://github.com/NVIDIA/dgx-selinux/)

To check what CUDA version do you need, you can either run `nvidia-smi` or `nvcc --version`.

Alternatively, you can also check nvidia-smi with docker:

```
docker run --runtime=nvidia --rm nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

To use CUDA, use the images with the `cublas` tag, for example.

The image list is on [quay](https://quay.io/repository/go-skynet/local-ai?tab=tags):

- CUDA `11` tags: `master-gpu-nvidia-cuda-11`, `v1.40.0-gpu-nvidia-cuda-11`, ...
- CUDA `12` tags: `master-gpu-nvidia-cuda-12`, `v1.40.0-gpu-nvidia-cuda-12`, ...
- CUDA `13` tags: `master-gpu-nvidia-cuda-13`, `v1.40.0-gpu-nvidia-cuda-13`, ...

In addition to the commands to run LocalAI normally, you need to specify `--gpus all` to docker, for example:

```bash
docker run --rm -ti --gpus all -p 8080:8080 -e DEBUG=true -e MODELS_PATH=/models -e THREADS=1 -v $PWD/models:/models quay.io/go-skynet/local-ai:v1.40.0-gpu-nvidia-cuda12
```

If the GPU inferencing is working, you should be able to see something like:

```
5:22PM DBG Loading model in memory from file: /models/open-llama-7b-q4_0.bin
ggml_init_cublas: found 1 CUDA devices:
  Device 0: Tesla T4
llama.cpp: loading model from /models/open-llama-7b-q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 1024
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 2 (mostly Q4_0)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =    0.07 MB
llama_model_load_internal: using CUDA for GPU acceleration
llama_model_load_internal: mem required  = 4321.77 MB (+ 1026.00 MB per state)
llama_model_load_internal: allocating batch_size x 1 MB = 512 MB VRAM for the scratch buffer
llama_model_load_internal: offloading 10 repeating layers to GPU
llama_model_load_internal: offloaded 10/35 layers to GPU
llama_model_load_internal: total VRAM used: 1598 MB
...................................................................................................
llama_init_from_file: kv self size  =  512.00 MB
```

## ROCm (AMD) acceleration

There are a limited number of tested configurations for ROCm systems however most newer dedicated GPU consumer grade devices seem to be supported under the current ROCm 6 implementation.

Due to the nature of ROCm it is best to run all implementations in containers as this limits the number of packages required for installation on the host system. Compatibility and package versions for dependencies across all variations of OS must be tested independently if desired; please refer to the [build]({{%relref "installation/build#Acceleration" %}}) documentation.

### Hardware requirements and compatibility

- **GPU**: ROCm 6.x.x compatible GPU or accelerator
- **Architecture targets**: LocalAI hipblas images are pre-built for the following LLVM targets: `gfx900`, `gfx906`, `gfx908`, `gfx940`, `gfx941`, `gfx942`, `gfx90a`, `gfx1030`, `gfx1031`, `gfx1100`, `gfx1101`
- **OS**: Ubuntu (22.04, 24.04), RHEL (9.3, 9.2, 8.9, 8.8), SLES (15.5, 15.4)
- **Host packages**: `amdgpu-dkms` and `rocm` >= 6.0.0
- **Disk space**: At least 100GB free on the disk hosting the container runtime (ROCm images can be ~20GB)

Common AMD GPU models and their LLVM targets:

| GPU | LLVM Target |
| --- | --- |
| Radeon VII | gfx906 |
| RX 6800 / 6900 XT | gfx1030 |
| RX 7900 XTX / 7900 XT | gfx1100 |
| RX 7800 XT / 7700 XT | gfx1101 |
| MI100 | gfx908 |
| MI210 / MI250 | gfx90a |
| MI300X | gfx942 |

If your device is not in the pre-built target list, you must set `GPU_TARGETS` and `REBUILD=true` as environment variables.

### ROCm installation steps

1. Check your GPU LLVM target is compatible with the version of ROCm. This can be found in the [LLVM Docs](https://llvm.org/docs/AMDGPUUsage.html).
2. Check which ROCm version is compatible with your LLVM target and your chosen OS (pay special attention to supported kernel versions). See the following for compatibility for [ROCm 6.0.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/reference/system-requirements.html) or [ROCm 6.0.2](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).
3. Install `amdgpu-dkms` using your native package manager, then **reboot** the system.
4. Install `rocm` using your native package manager. See the installation documentation for [6.0.2](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html) or [6.0.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/how-to/native-install/index.html).
5. Verify the installation:

```bash
# Check that the GPU is detected
rocm-smi

# Verify ROCm version
cat /opt/rocm/.info/version
```

### Docker container setup with ROCm

Docker compose example:

```yaml
    image: quay.io/go-skynet/local-ai:master-aio-gpu-hipblas
    environment:
      - DEBUG=true
      # Only needed if your GPU target is not in the pre-built list
      - REBUILD=true
      - BUILD_TYPE=hipblas
      - GPU_TARGETS=gfx906 # Example for Radeon VII
    devices:
      - /dev/dri
      - /dev/kfd
```

Docker run command:

```bash
docker run \
  -p 8080:8080 \
  -e DEBUG=true \
  -v $PWD/models:/models \
  --device /dev/dri \
  --device /dev/kfd \
  quay.io/go-skynet/local-ai:master-aio-gpu-hipblas
```

If your GPU requires a custom target:

```bash
docker run \
  -p 8080:8080 \
  -e DEBUG=true \
  -e REBUILD=true \
  -e BUILD_TYPE=hipblas \
  -e GPU_TARGETS=gfx1100 \
  -v $PWD/models:/models \
  --device /dev/dri \
  --device /dev/kfd \
  quay.io/go-skynet/local-ai:master-aio-gpu-hipblas
```

The rebuild process will take some time to complete. It is recommended that you `pull` the image prior to deployment.

### Environment variables

| Variable | Description | Example |
| --- | --- | --- |
| `BUILD_TYPE` | Set to `hipblas` for AMD GPU support | `hipblas` |
| `GPU_TARGETS` | LLVM target for your GPU | `gfx1100` |
| `REBUILD` | Forces rebuild with custom GPU targets | `true` |
| `HIP_VISIBLE_DEVICES` | Controls which GPUs are visible to the runtime | `0`, `0,2` |
| `ROCR_VISIBLE_DEVICES` | Alternative device visibility control | `0`, `0,1` |

### Kubernetes deployment

For k8s deployments, deploy the [ROCm/k8s-device-plugin](https://artifacthub.io/packages/helm/amd-gpu-helm/amd-gpu) first. If you use rke2 or OpenShift, deploy the SUSE or RedHat provided version for compatibility.

After the device plugin is installed, the [helm chart from go-skynet](https://github.com/go-skynet/helm-charts) can be configured with the following GPU-relevant settings:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {NAME}-local-ai
...
spec:
  ...
  template:
    ...
    spec:
      containers:
        - env:
            - name: HIP_VISIBLE_DEVICES
              value: '0'
              # 0:device1 1:device2 2:device3 etc.
              # For multiple devices (say device 1 and 3): HIP_VISIBLE_DEVICES="0,2"
          ...
          resources:
            limits:
              amd.com/gpu: '1'
            requests:
              amd.com/gpu: '1'
```

### Verified backends

The devices in the following list have been tested with `hipblas` images running ROCm 6.0.0:

| Backend | Verified | Devices |
| ---- | ---- | ---- |
| llama.cpp | yes | Radeon VII (gfx906) |
| diffusers | yes | Radeon VII (gfx906) |
| piper | yes | Radeon VII (gfx906) |
| whisper | no | none |
| coqui | no | none |
| transformers | no | none |
| sentencetransformers | no | none |
| transformers-musicgen | no | none |
| vllm | no | none |

**You can help by expanding this list.**

### Troubleshooting AMD GPU issues

**GPU not detected in container:**

```bash
# Verify host can see the GPU
rocm-smi

# Check device permissions
ls -la /dev/dri /dev/kfd

# Ensure your user is in the 'render' and 'video' groups
sudo usermod -a -G render,video $USER
```

**Wrong GPU target errors:** If you see errors like `hipErrorNoBinaryForGpu`, your GPU target is not included in the pre-built image. Set `GPU_TARGETS` and `REBUILD=true`.

**ROCm version mismatch:** Ensure the host ROCm driver version is equal to or newer than the version used in the LocalAI image (6.0.0 at time of writing).

**`Error 413` on file upload (k8s):** The ingress may require the annotation `nginx.ingress.kubernetes.io/proxy-body-size: "25m"` to allow larger uploads.

### Recommendations

- Do not use the GPU assigned for compute for desktop rendering.
- When installing the ROCm kernel driver, install an equal or newer version than what is implemented in LocalAI.

## Intel acceleration (SYCL/oneAPI)

### Hardware requirements

- **Supported GPUs**: Intel Arc A-series (A770, A750, A580, A380), Intel Data Center GPU Max Series, Intel Data Center GPU Flex Series, Intel integrated GPUs (various generations)
- **OS**: Ubuntu 22.04 or 24.04 (recommended), other Linux distributions with Intel GPU driver support
- **Drivers**: Intel GPU drivers must be installed on the host system

### oneAPI installation (building from source)

If building from source rather than using containers, install the [Intel oneAPI Base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html) and ensure Intel GPU drivers are available in the system.

```bash
# Install Intel GPU drivers
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero

# Install oneAPI Base Toolkit (follow Intel's official guide for your OS)
# After installation, source the environment:
source /opt/intel/oneapi/setvars.sh
```

### SYCL backend configuration

LocalAI supports multiple SYCL build types:

| Build Type | Description |
| --- | --- |
| `intel` | Default Intel SYCL build |
| `sycl_f16` | SYCL with FP16 support (better performance on supported hardware) |
| `sycl_f32` | SYCL with FP32 only |

When building from source, set the `BUILD_TYPE` accordingly:

```bash
BUILD_TYPE=intel make build
```

### Container images

To use SYCL, use images with `gpu-intel` in the tag, for example `{{< version >}}-gpu-intel`.

The image list is on [quay](https://quay.io/repository/go-skynet/local-ai?tab=tags).

The base image used for Intel builds is `intel/oneapi-basekit:2025.3.0-0-devel-ubuntu24.04`.

### Docker setup for Intel GPUs

Basic run command:

```bash
docker run --rm -ti --device /dev/dri \
  -p 8080:8080 \
  -e DEBUG=true \
  -e MODELS_PATH=/models \
  -e THREADS=1 \
  -v $PWD/models:/models \
  quay.io/go-skynet/local-ai:{{< version >}}-gpu-intel
```

To run with a specific model (e.g., `phi-2`):

```bash
docker run -e DEBUG=true --privileged -ti \
  -v $PWD/models:/models \
  -p 8080:8080 \
  -v /dev/dri:/dev/dri \
  --rm quay.io/go-skynet/local-ai:master-gpu-intel phi-2
```

Docker compose example:

```yaml
services:
  localai:
    image: quay.io/go-skynet/local-ai:master-gpu-intel
    ports:
      - "8080:8080"
    environment:
      - DEBUG=true
      - MODELS_PATH=/models
      - THREADS=1
    devices:
      - /dev/dri:/dev/dri
    volumes:
      - ./models:/models
```

### Performance tuning tips

- Set `mmap: false` in your model configuration. SYCL has a known issue that causes hangs with `mmap: true`.
- Use `f16: true` in model configs when using `sycl_f16` builds for better performance.
- Set `THREADS` to 1 when offloading fully to the GPU.
- For Intel Arc GPUs, ensure you are using the latest driver version for optimal performance.
- Monitor GPU utilization with `intel_gpu_top` (part of the `intel-gpu-tools` package).

### Verifying Intel GPU detection

```bash
# Check Intel GPU is visible
ls /dev/dri

# List Intel GPUs (requires intel-gpu-tools)
sudo intel_gpu_top

# Check OpenCL device availability
clinfo | grep "Device Name"

# Inside the container, verify SYCL device detection
sycl-ls
```

## NVIDIA L4T (Jetson) acceleration

LocalAI can run on NVIDIA ARM64 devices using the L4T (Linux for Tegra) platform. For the full reference, see the [Nvidia ARM64]({{%relref "reference/nvidia-l4t" %}}) documentation.

### Jetson hardware compatibility

| Platform | CUDA Version | Base Image |
| --- | --- | --- |
| Jetson AGX Orin | CUDA 12 | `nvcr.io/nvidia/l4t-jetpack:r36.4.0` |
| Jetson Orin NX / Nano | CUDA 12 | `nvcr.io/nvidia/l4t-jetpack:r36.4.0` |
| Jetson Xavier NX / AGX Xavier | CUDA 12 | `nvcr.io/nvidia/l4t-jetpack:r36.4.0` |
| NVIDIA DGX Spark | CUDA 13 | `ubuntu:24.04` |

### Prerequisites

- Docker engine installed ([installation guide](https://docs.docker.com/engine/install/ubuntu/))
- NVIDIA container toolkit installed ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- JetPack SDK installed on the Jetson device

### Pre-built images

CUDA 12 (AGX Orin and similar):

```bash
docker pull quay.io/go-skynet/local-ai:latest-nvidia-l4t-arm64
```

CUDA 13 (DGX Spark):

```bash
docker pull quay.io/go-skynet/local-ai:latest-nvidia-l4t-arm64-cuda-13
```

### Docker setup

```bash
# CUDA 12 (Jetson Orin)
docker run -e DEBUG=true -p 8080:8080 \
  -v /data/models:/models \
  -ti --restart=always \
  --name local-ai \
  --runtime nvidia --gpus all \
  quay.io/go-skynet/local-ai:latest-nvidia-l4t-arm64

# CUDA 13 (DGX Spark)
docker run -e DEBUG=true -p 8080:8080 \
  -v /data/models:/models \
  -ti --restart=always \
  --name local-ai \
  --runtime nvidia --gpus all \
  quay.io/go-skynet/local-ai:latest-nvidia-l4t-arm64-cuda-13
```

### Building from source for L4T

If the pre-built images don't meet your needs:

```bash
git clone https://github.com/mudler/LocalAI
cd LocalAI

# CUDA 12 (Orin)
docker build \
  --build-arg SKIP_DRIVERS=true \
  --build-arg BUILD_TYPE=cublas \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  --build-arg IMAGE_TYPE=core \
  -t localai:l4t-arm64 .

# CUDA 13 (DGX Spark)
docker build \
  --build-arg SKIP_DRIVERS=false \
  --build-arg BUILD_TYPE=cublas \
  --build-arg CUDA_MAJOR_VERSION=13 \
  --build-arg CUDA_MINOR_VERSION=0 \
  --build-arg BASE_IMAGE=ubuntu:24.04 \
  --build-arg IMAGE_TYPE=core \
  -t localai:l4t-arm64-cuda13 .
```

### Performance considerations for edge devices

- **Memory**: Jetson devices share memory between CPU and GPU. Use smaller quantized models (Q4_0, Q4_K_M) to fit within available memory.
- **GPU layers**: Start with fewer `gpu_layers` and increase gradually while monitoring memory usage with `tegrastats`.
- **Threads**: Set `threads: 1` for GPU-offloaded workloads to avoid CPU contention.
- **Swap**: Configure swap space if running larger models. Jetson Orin supports NVMe-backed swap for better performance.
- **Power mode**: Use `nvpmodel` to set the appropriate power mode for your workload:

```bash
# List available power modes
sudo nvpmodel -q

# Set max performance mode (varies by device)
sudo nvpmodel -m 0

# Maximize clock speeds
sudo jetson_clocks
```

## Vulkan acceleration

### Requirements

If using nvidia, follow the steps in the [CUDA](#cuda-nvidia-acceleration) section to configure your docker runtime to allow access to the GPU.

### Container images

To use Vulkan, use the images with the `vulkan` tag, for example `{{< version >}}-gpu-vulkan`.

#### Example

To run LocalAI with Docker and Vulkan, you can use the following command as an example:

```bash
docker run -p 8080:8080 -e DEBUG=true -v $PWD/models:/models localai/localai:latest-gpu-vulkan
```

### Notes

In addition to the commands to run LocalAI normally, you need to specify additional flags to pass the GPU hardware to the container.

These flags are the same as the sections above, depending on the hardware, for [NVIDIA](#cuda-nvidia-acceleration), [AMD](#rocm-amd-acceleration) or [Intel](#intel-acceleration-sycloneapi).

If you have mixed hardware, you can pass flags for multiple GPUs, for example:

```bash
docker run -p 8080:8080 -e DEBUG=true -v $PWD/models:/models \
--gpus=all \ # nvidia passthrough
--device /dev/dri --device /dev/kfd \ # AMD/Intel passthrough
localai/localai:latest-gpu-vulkan
```

## Multi-GPU configuration

### Tensor parallelism with diffusers

For multi-GPU support with diffusers, configure the model with `tensor_parallel_size` set to the number of GPUs:

```yaml
name: stable-diffusion-multigpu
model: stabilityai/stable-diffusion-xl-base-1.0
backend: diffusers
parameters:
  tensor_parallel_size: 2 # Number of GPUs to use
```

When `tensor_parallel_size` is set to a value greater than 1, the diffusers backend automatically enables `device_map="auto"` to distribute the model across multiple GPUs.

### Device selection and assignment

Control which GPUs are visible to LocalAI using vendor-specific environment variables:

```bash
# NVIDIA: select specific GPUs by index
CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all ...

# AMD: select specific GPUs by index
HIP_VISIBLE_DEVICES=0,2 docker run --device /dev/dri --device /dev/kfd ...

# Intel: select specific device
ONEAPI_DEVICE_SELECTOR="level_zero:0" docker run --device /dev/dri ...
```

### llama.cpp multi-GPU with tensor splitting

For llama.cpp models, you can distribute layers across GPUs using `gpu_layers` along with `tensor_split` to control the distribution ratio:

```yaml
name: my-large-model
parameters:
  model: large-model.gguf
gpu_layers: 99
# Split ratio between GPUs (proportional to VRAM)
# Example: 60% on GPU 0, 40% on GPU 1
tensor_split: "0.6,0.4"
```

### Distributed inference across nodes

For workloads that exceed the capacity of a single machine, LocalAI supports distributed inference. See the [Distributed Inference]({{%relref "features/distributed_inferencing" %}}) documentation for details on worker mode and federated mode.

### Performance tuning for multi-GPU

- Use GPUs of the same type and memory capacity for optimal performance.
- Ensure sufficient GPU memory across all devices before loading a model.
- Set `threads: 1` when using GPU acceleration to avoid CPU-GPU synchronization overhead.
- Monitor per-GPU utilization to identify bottlenecks (see [GPU Monitoring](#gpu-monitoring)).
- For llama.cpp, set `tensor_split` proportional to VRAM available on each GPU if GPUs are asymmetric.

### Common pitfalls

- **Mixing GPU types**: Different GPU architectures in the same tensor-parallel setup can cause significant performance degradation due to the slowest GPU bottlenecking the computation.
- **PCIe bandwidth**: Multi-GPU communication relies on PCIe or NVLink. PCIe 3.0 x16 systems may see limited scaling beyond 2 GPUs.
- **VRAM fragmentation**: Loading and unloading multiple models can fragment VRAM. Restart the container if you observe unexplained out-of-memory errors.
- **iGPU interference**: If your system has an integrated GPU, it may be enumerated as device 0. Use device visibility variables to exclude it.

## GPU monitoring

### NVIDIA (`nvidia-smi`)

```bash
# One-time status check
nvidia-smi

# Continuous monitoring (updates every 1 second)
nvidia-smi -l 1

# Query specific metrics
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1

# Monitor specific processes
nvidia-smi pmon -i 0 -s u -d 1
```

### AMD (`rocm-smi`)

```bash
# One-time status check
rocm-smi

# Show detailed GPU info
rocm-smi --showallinfo

# Continuous monitoring with temperature, utilization, and memory
watch -n 1 rocm-smi

# Show memory usage
rocm-smi --showmeminfo vram

# Show GPU utilization
rocm-smi --showuse
```

### Intel (`intel_gpu_top`)

```bash
# Install Intel GPU tools
sudo apt-get install intel-gpu-tools

# Real-time GPU monitoring
sudo intel_gpu_top

# List SYCL devices
sycl-ls

# Check OpenCL device info
clinfo
```

### NVIDIA Jetson (`tegrastats`)

```bash
# Real-time monitoring for Jetson devices
tegrastats

# Log to file with interval (milliseconds)
tegrastats --interval 1000 --logfile gpu_stats.log
```

### Monitoring inside Docker

When running LocalAI in a container, run monitoring tools on the host:

```bash
# NVIDIA - works from host, shows container GPU processes
nvidia-smi

# AMD - works from host
rocm-smi

# Check LocalAI logs for GPU-related messages
docker logs <container_id> 2>&1 | grep -i -E "gpu|cuda|rocm|sycl|vram"
```

### LocalAI debug mode

Enable debug logging for detailed GPU information:

```bash
docker run -e DEBUG=true ... quay.io/go-skynet/local-ai:...
```

With `DEBUG=true`, LocalAI logs GPU detection, memory allocation, and layer offloading details during model loading.

## Common issues and troubleshooting

### Driver compatibility

**Symptom**: Container fails to start or GPU is not detected.

```bash
# NVIDIA: verify driver and CUDA compatibility
nvidia-smi
# The output shows the driver version and maximum supported CUDA version.
# Ensure the LocalAI image CUDA version does not exceed your driver's supported version.

# AMD: verify ROCm driver
dkms status | grep amdgpu
cat /opt/rocm/.info/version

# Intel: verify GPU drivers
ls /dev/dri/render*
```

**Fix**: Update GPU drivers to a version compatible with the LocalAI image you are using. For NVIDIA, the driver must support the CUDA version used in the image. For AMD, the host ROCm version should be equal to or newer than the container's.

### Memory allocation errors

**Symptom**: `CUDA out of memory`, `HIP out of memory`, or model fails to load.

- Reduce `gpu_layers` in your model configuration to offload fewer layers to the GPU.
- Use a smaller quantized model (e.g., Q4_0 instead of Q8_0).
- Enable VRAM management with `--max-active-backends=1` to ensure only one model is loaded at a time. See [VRAM Management]({{%relref "advanced/vram-management" %}}).
- Monitor VRAM usage to right-size your configuration:

```bash
# Check available VRAM before loading
nvidia-smi --query-gpu=memory.free --format=csv  # NVIDIA
rocm-smi --showmeminfo vram                       # AMD
```

### Performance degradation

**Symptom**: GPU inference is slower than expected, or GPU utilization is low.

- Ensure `threads` is set to `1` in your model config when using GPU acceleration. Higher thread counts can cause CPU-GPU synchronization overhead.
- Set `f16: true` for GPU-accelerated models.
- Verify the model is actually using the GPU by checking logs with `DEBUG=true`.
- For llama.cpp, increase `gpu_layers` to offload more computation to the GPU. Setting it to `99` or higher offloads all layers.
- Check that no other processes are competing for GPU resources.

### Backend initialization failures

**Symptom**: `failed to load backend`, `cannot find GPU`, or backend crashes on startup.

```bash
# Check LocalAI logs for specific errors
docker logs <container_id> 2>&1 | tail -50

# Verify the correct image is being used for your hardware
# NVIDIA: use *-gpu-nvidia-cuda-* images
# AMD: use *-gpu-hipblas images
# Intel: use *-gpu-intel images
```

- Ensure you are using the correct image tag for your GPU vendor and CUDA/ROCm version.
- For AMD GPUs with non-standard targets, set `REBUILD=true` and `GPU_TARGETS` to your device's LLVM target.
- For Intel SYCL, ensure `/dev/dri` is passed to the container.

### Container GPU passthrough problems

**NVIDIA**: Ensure `nvidia-container-toolkit` is installed and the Docker runtime is configured:

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# If this fails, configure the runtime:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**AMD**: Ensure `/dev/dri` and `/dev/kfd` are passed to the container, and the user has the correct group permissions:

```bash
# Check device permissions
ls -la /dev/dri /dev/kfd

# Add user to required groups
sudo usermod -a -G render,video $USER
# Log out and back in for group changes to take effect
```

**Intel**: Ensure `/dev/dri` is passed to the container:

```bash
# Verify render nodes exist
ls /dev/dri/render*

# Pass the device to Docker
docker run --device /dev/dri ...
```

**SELinux**: If running on a system with SELinux (e.g., RHEL, Fedora), you may need to set appropriate SELinux policies. For NVIDIA, see the [dgx-selinux policies](https://github.com/NVIDIA/dgx-selinux/). Alternatively, test with `--security-opt label=disable` to confirm SELinux is the issue.

### Checking GPU usage within LocalAI

Use the LocalAI metrics endpoint to monitor backend status:

```bash
# Check loaded backends and their status
curl http://localhost:8080/metrics

# Manually stop a model to free GPU memory
curl -X POST http://localhost:8080/backend/shutdown \
  -H "Content-Type: application/json" \
  -d '{"model": "model-name"}'
```

## Related documentation

- [VRAM and Memory Management]({{%relref "advanced/vram-management" %}}) - Automatic GPU memory management with LRU eviction and watchdogs
- [Distributed Inference]({{%relref "features/distributed_inferencing" %}}) - Multi-node inference with worker and federated modes
- [Running on Nvidia ARM64]({{%relref "reference/nvidia-l4t" %}}) - Full L4T reference guide
- [Build from Source]({{%relref "installation/build#Acceleration" %}}) - Building LocalAI with GPU acceleration support
