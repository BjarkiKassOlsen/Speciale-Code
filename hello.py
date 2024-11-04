
from project_imports import *

#describe the environment that has been loaded:
print("Hello World!")
print(DATA_PATH)

print('The name of my current conda environment is:')
print(os.environ['CONDA_DEFAULT_ENV'])
model_run_nr = int(os.getenv('SLURM_ARRAY_TASK_ID'))
print(model_run_nr)
 
print(model_run_nr+2)
