This project fits simulated X-ray binary spectra using several spectral models, including:

- Power law
- Blackbody
- Disk blackbody (`diskbb`)
- Composite power law + diskbb models

The repository contains fitting functions and a Jupyter notebook used to generate fits and plots.

----

To run the files

Install the required Python packages:

`import numpy as np
import matplotlib.pyplot as plt
import athena_mc as athenamc
from scipy.optimize import curve_fit
from scipy.integrate import quad
import final_proj as fitting`

Open the notebook `final_project.ipynb` and run all cells. This will run both of the test runs using the functions in the `final_proj.py` file.