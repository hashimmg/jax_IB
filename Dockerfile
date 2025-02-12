FROM nvidia/cuda:12.6.1-base-ubuntu24.04

WORKDIR /tmp/
RUN apt-get update && apt-get install -y curl gpg wget bzip2 git build-essential

# copy over local jax_ib version
RUN mkdir -p /tmp/jax_ib/jax_ib
RUN mkdir -p /tmp/jax_ib/scripts
COPY jax_ib /tmp/jax_ib/jax_ib
COPY setup.py /tmp/jax_ib/setup.py
COPY docker/environment.yml environment.yml

# setup micromamba venv manager
ENV MAMBA_ROOT_PREFIX="/tmp/micromamba"
ENV MAMBA_EXE="/bin/micromamba"
RUN mkdir -p ${MAMBA_ROOT_PREFIX}
COPY docker/setup_micromamba.sh setup_micromamba.sh
RUN ./setup_micromamba.sh

# create micromamba environment
RUN micromamba create -y --name jax-cfd python==3.12 --file environment.yml

# initialize bash
RUN micromamba shell init --shell bash --root-prefix=${MAMBA_ROOT_PREFIX}

# we'll use an interactive shell to run the application
# always activate the environment in interactive shell
RUN echo "micromamba activate jax-cfd" >> ~/.bashrc

# copy over the application scrips
COPY scripts/flap-opt.py /tmp/jax_ib/scripts/flap-opt.py
COPY scripts/config.yml /tmp/jax_ib/scripts/config.yml

# create the entrypoint for docker run
COPY docker/entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

# set the workdir
WORKDIR /tmp/jax_ib/scripts

ENTRYPOINT ["/bin/bash","-i","/tmp/entrypoint.sh"]
CMD ["/bin/bash", "-i"]
