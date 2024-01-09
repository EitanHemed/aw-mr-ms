
<h1 style="text-align: center; font-weight: bold">Manuscript: Mental rotation of features and conjunctions</h1>

### Corresponding author - [Assaf Weksler](assaf.weksler@gmail.com)


#### For technical issues - please [open an issue](https://github.com/EitanHemed/mr-colour/issues) or contact [Eitan Hemed](Eitan.Hemed@gmail.com)


---
## Repository Structure

The repository is structured as follows:
1. [Data Analysis](<Data Analysis>) - contains all the code used for the data analysis and figure generation.
 
5. [Docker](<Docker>) - files related to the preperation of the docker image, enabling reproducibility of analyses.

---

## Setup 

There are two options for setting up the environment for replication: docker and local installation.

We recommend using the docker image, as it is easier to set up, run and remove afterwards.
Also, this is less likely to interfere with your local Python and R installations.

#### Option 1: Docker

First, follow this guide to install Docker on your system: https://docs.docker.com/get-docker/

After setting up Docker on your system, pull the image from Docker Hub by running the following command in the terminal:

`docker pull eitanhemed/aw-mr-ms:latest`

Then, run the following command to start the container, which will also launch up a Jupyter server:

`docker run -p 8888:8888 eitanhemed/aw-mr-ms`

Once the Jupyter server is up, you can access it by opening the following link in your browser, 
for example by going to the terminal and clicking on the link: `http://127.0.0.1:8888/tree`, or any other link that 
appears in your terminal. 


#### Option 2: Local Installation

To install locally, first make sure you have git and conda installed on your system.

Anaconda can be downloaded from here: https://www.anaconda.com/download

Then, run the following commands in the terminal:
```
git clone https://github.com/EitanHemed/aw-mr-ms.git
cd aw-mr-ms-master
conda env create -f './Data Analysis/env.yml'
```

This should create an environment called `aw_mr` with all the necessary packages installed.

Take note that as [robusta](https://github.com/EitanHemed/robusta) requires an R installation,
it could take a couple of minutes to install all the packages (and up to 20-30 minutes on Linux).

Hence, our recommendation is to use the docker image.

---

## Reproduction

#### Option 1: Docker 
If you are using Docker, then when running the container, you will be prompted with a link to the Jupyter server (e.g., 
http://127.0.0.1:8888/tree).

Then, on the root folder you will view a notebook named `reproduc-nb.ipynb`, which contains all the relevant calls.
The relevant conda environment is already activated, so you can just run the notebook (or just browse the files).

#### Option 2: Local Installation
If you are using a local installation, then first activate the environment by activating the environment:
`conda activate aw_mr`

Then, run the following commands in the terminal, or copy them to a notebook:
```
cd './Data Analysis'
python run_pipeline.py
```
or in a Python session:
```python
import run_pipeline
run_pipeline.main()
```

Note that you can also view the notebook [Docker/reproduc-nb.ipynb](Docker/reproduc-nb.ipynb). Just make sure to copy
it to the root folder of the project if using the local installation option.

