# prog-econ-final-project
Final project for the course "Effective Programming Practices for Economists"

# Goal

- Visualize feature importance based on [LightGBM](https://lightgbm.readthedocs.io/en/latest/) framework's learning algorithm.

- Conduct feature selection and compare its performance

**Disclaimer:** I already provided the parameters for LGBM in this repo, since it takes a lot of
time to optimize them. If you want to optimize them yourself, you can run script 
*parameter_visualization.py* in folder *code*.

# Usage
## Get the data
Since the data is to big to be stored in github, you can download it from [my OneDrive](https://1drv.ms/u/s!Asz7v_UERQ36oh1BI3nPKjprKqO8).  

## Run
Just run main.py

However, since the process takes very long with the whole data,
one can decrease the number of rows by given the variable "num_rows" the respective integer value.
Currently, its value is set to 20000:

<p align="center"><img src="https://i.imgur.com/uUKT8vp.png"></p>

# Data
<p align="center"><img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png"></p>
