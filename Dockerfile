FROM quay.io/pypa/manylinux2014_x86_64
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh --output /miniconda.sh
RUN sh /miniconda.sh -b -p /anaconda
RUN source /anaconda/etc/profile.d/conda.sh
RUN /anaconda/condabin/conda install -y -c conda-forge -n base suitesparse pybind11
RUN /anaconda/condabin/conda activate base
