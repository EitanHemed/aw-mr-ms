FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
RUN git clone https://github.com/EitanHemed/aw-mr-ms

RUN conda update -n base -c defaults conda

RUN conda create --name aw_mr python=3.9.12 pip jupyter matplotlib=3.7.1 seaborn=0.12.2 pandas=1.2.5 r-base=4.1.3 gcc r-broom r-afex -y --quiet -c conda-forge

# Make RUN commands use the new environment:
RUN echo "conda activate aw_mr" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

EXPOSE 8888

# The code to run when container is started:
COPY entrypoint.sh reproduc-nb.ipynb .

# Make entrypoint.sh executable
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

