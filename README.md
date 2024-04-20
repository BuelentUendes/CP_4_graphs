# Conformal Predictions for Graphs
Code basis for the final project of the course: **"Machine Learning for Graphs"** taught at the Vrije Universiteit Amsterdam, 2024. 
In this repo, I implement, compare and reproduce results for different conformal prediction methods for various graph machine learning models. In particular,
I reproduce the methods introduced in the paper "Conformal Prediction Sets for Graph Neural Networks" which can be found [here](https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf)
and therefore this repo heavily depends upon their [code bases](https://github.com/soroushzargar/DAPS).

**This is work in progress and will be updated incrementally.**
___
**Installation:**

1. Create a virtualenv of your choice

~~~
$ virtualenv <NAME_OF_YOUR_VIRTUALENV> 
~~~

2. Activate the virtual env
~~~
$ source <NAME_OF_YOUR_VIRTUALENV>/bin/activate
~~~

3. Install the required packages for this project

~~~
pip3 install -r requirements.txt
~~~

4. Now you can check if the installation has worked by running:

~~~
python3 train_models.py
~~~

The "src" folder includes the source code for the models, whereas the "utils" folder includes additional utility functions needed for the project.

**Tested environment:**

The project was run and tested on both MacOS and Linux Ubuntu with Python 3.9 installed.


