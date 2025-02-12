FROM nvidia/cuda:12.6.1-base-ubuntu24.04

WORKDIR /tmp/
RUN apt-get update && apt-get install -y curl gpg wget bzip2 git build-essential
RUN mkdir /scratch
RUN mkdir -p /scratch/jax_ib/jax_ib
RUN mkdir -p /scratch/jax_ib/scripts
COPY jax_ib /scratch/jax_ib/jax_ib

COPY setup.py /scratch/jax_ib/setup.py
COPY environment.yml environment.yml


ENV MAMBA_ROOT_PREFIX="/scratch/micromamba"
ENV MAMBA_EXE="/bin/micromamba"
RUN mkdir -p ${MAMBA_ROOT_PREFIX}
ADD setup_micromamba.sh setup_micromamba.sh
RUN ./setup_micromamba.sh
RUN micromamba create -y --name jax-cfd python==3.12 --file environment.yml
RUN micromamba shell init --shell bash --root-prefix=~/micromamba
ADD dockerfile_shell.sh /tmp/dockerfile_shell.sh
RUN echo "micromamba activate jax-cfd" >> ~/.bashrc
RUN sed -e '/[ -z "$PS1" ] && return/s/^/#/g' -i ~/.bashrc
SHELL ["/tmp/dockerfile_shell.sh"]
COPY notebooks/flap-opt.py /scratch/jax_ib/scripts/flap-opt.py
COPY notebooks/config.yml /scratch/jax_ib/scripts/config.yml
COPY entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh
WORKDIR /scratch/jax_ib/scripts
ENTRYPOINT ["/bin/bash","-l","/tmp/entrypoint.sh"]


