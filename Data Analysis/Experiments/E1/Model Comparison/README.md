To reproduce the Bayesian ANOVA reported in the paper (for the current dataset or for another) - 
1. install version `0.17.2.1` of [JASP](https://github.com/jasp-stats/jasp-desktop/releases/tag/v0.17.2.1).
2. Load a version of the data from the `Output/Analyses/X/Data/pivot_aggregated_task.csv` into JASP (where `X` stands for the version
of screening parameters you want to use).
3. Select `ANOVA>Bayesian Repeated Measures ANOVA` from the drop-down menu.
4. Enter the relevant variables.
5. Under `Model` options below select Rotation Size and Chromaticity as nuisance.
6. Under `Additional Options` Set the seed to 42. 

