#!/bin/bash

# Remove old output
rm -rf *.nvprof


yes | pkilluser `whoami`

# Release nvprof 
pexec "rm -rf /tmp/.nvprof/nvprof.lock"

# Print GPU status from the head and execute from nodes
pexec "nvidia-smi"

# Launch nvprof
pexec "nvprof --profile-all-processes --output-profile output.%h.%p.nvprof"
ctest -R Test_Hybrid_HeatFlow*
# ctest -R Test_Hybrid_HeatFlow_Strong_4_processes

 
yes | pkilluser `whoami`

