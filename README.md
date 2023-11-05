
<h1 style="text-align: center; font-weight: bold">Manuscript: The mental rotation of visual features and feature conjunctions</h1>

### Corresponding author - [Assaf Weksler](assaf.weksler@gmail.com)
#### For technical issues - please [open an issue](https://github.com/EitanHemed/mr-colour/issues) or contact [Eitan Hemed](Eitan.Hemed@gmail.com)

---

## Abstract

---
## Repository Structure

[Data Analysis](<Data Analysis>) - contains all the code used for the data analysis and figure generation.

---

## Setup

To install locally, **first make sure you have git and conda installed** on your system. 

Then, run the following commands in the terminal:
```
git clone https://github.com/EitanHemed/mr-colour.git
conda env create -f './Data Analysis/env.yml'
```
This will create an environment called `mr-ms` with all the necessary packages installed.

Take note that as [robusta](https://github.com/EitanHemed/robusta) requires an R installation,
it could take a couple of minutes to install all the packages (especially on Linux). 

---

## Usage
To reproduce the analysis, you need to activate the new conda environment and run the `run_pipeline.py` script from the 
`Data Analysis` directory. This will run all the analysis and generate the figures.

```
conda activate mr_ms
cd './Data Analysis'
python run_pipeline.py
```




