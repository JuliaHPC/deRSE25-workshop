# syntax=docker/dockerfile:1
FROM julia:1.11.3

# Install git, for use within Codespaces
RUN /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y git \
    && apt-get --purge autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*'

# Docker is awful and doesn't allow conditionally setting environment variables in a decent
# way, so we have to keep an external script and source it every time we need it.
COPY julia_cpu_target.sh /julia_cpu_target.sh

RUN julia --color=yes -e 'using InteractiveUtils; versioninfo()'

# Instantiate Julia project
RUN mkdir -p /root/.julia/environments/v1.11
COPY Project.toml  /root/.julia/environments/v1.11/Project.toml
RUN . /julia_cpu_target.sh && julia --color=yes -e 'using Pkg; Pkg.instantiate()'
# Preinstall some common packages across all notebooks
RUN . /julia_cpu_target.sh && julia --color=yes -e 'using Pkg; Pkg.install(["KernelAbstractions", "ThreadPinning"])'

# Copy notebooks
COPY introduction/intro.jl /root/introduction/intro.jl
COPY kernelabstractions/diffusion.jl /root/kernelabstractions/diffusion.jl
COPY kernelabstractions/diffusion_kernel.jl /root/kernelabstractions/diffusion_kernel.jl
COPY kernelabstractions/multi-backend.jl /root/kernelabstractions/multi-backend.jl
COPY multithreading/julia-multithreading.jl /root/multithreading/julia-multithreading.jl
