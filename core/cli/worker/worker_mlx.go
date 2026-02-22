package worker

import (
	"fmt"
	"os"
	"strings"

	cliContext "github.com/mudler/LocalAI/core/cli/context"
	"github.com/mudler/LocalAI/pkg/system"
)

type MLXRDMA struct {
	WorkerFlags `embed:""`
}

func (r *MLXRDMA) Run(ctx *cliContext.Context) error {
	if len(os.Args) < 4 {
		return fmt.Errorf("usage: local-ai worker mlx-rdma -- <mlx-rdma-args>")
	}

	systemState, err := system.GetSystemState(
		system.WithBackendPath(r.BackendsPath),
		system.WithBackendSystemPath(r.BackendsSystemPath),
	)
	if err != nil {
		return err
	}

	// Get the python binary
	pythonPath, err := system.GetPythonBinary(systemState)
	if err != nil {
		return err
	}

	// Get the backend path
	backendPath, err := getMLXBackendPath(systemState, r.BackendGalleries)
	if err != nil {
		return err
	}

	// Prepare the arguments
	args := strings.Split(r.ExtraMLXRDMAArgs, " ")
	args = append([]string{backendPath}, args...)

	// Set environment variables for RDMA
	if os.Getenv("MLX_GRPC_SERVERS") == "" {
		os.Setenv("MLX_GRPC_SERVERS", os.Getenv("LLAMACPP_GRPC_SERVERS"))
	}

	// Execute the backend
	return system.ExecPython(pythonPath, args, os.Environ())
}

func getMLXBackendPath(systemState *system.SystemState, galleries string) (string, error) {
	// TODO: Implement backend discovery for MLX (similar to llama.cpp)
	// For now, assume the backend is at a known location
	backend := "mlx"
	backendPath := systemState.BackendSystemPath

	// Check if backend exists
	fullPath := fmt.Sprintf("%s/%s/backend.py", backendPath, backend)
	if _, err := os.Stat(fullPath); err == nil {
		return fullPath, nil
	}

	// Fallback: try to find the backend in the system path
	return fmt.Sprintf("%s/backend.py", backendPath), nil
}