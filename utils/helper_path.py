# Helper file for the paths used in the project
import os
from os.path import abspath

# Define the common paths
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
MODELS_PATH = abspath(os.path.join(FILE_PATH, "./../", "saved_models"))
CONFIG_PATH = abspath(os.path.join(FILE_PATH, "./../", "configs"))
FIGURES_PATH = abspath(os.path.join(FILE_PATH, "./../", "figures"))
DATA_PATH = abspath(os.path.join(FILE_PATH, "./../", "datasets"))
RESULTS_PATH = abspath(os.path.join(FILE_PATH, "./../", "results"))
