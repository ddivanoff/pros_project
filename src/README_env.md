# How to set the environment 


## conda


In this case, make sure to include the following dependencies in the `environment.yml` file:


```
name: pros
dependencies:
 - kagglehub
 - ipykernel
 - ydata-profiling
 - ipywidgets
 - fastparquet
 - pyarrow
```


Then run in terminal `conda env create -f environment.yml`


To activate/deactivate:

`conda activate pros` / `conda deactivate`


## venv


In the desired dir, run in terminal: 


`python3 -m venv example-prj/example-venv`


To activate/deactivate:

`source example-prj/example-venv/bin/activate` / `deactivate`


Each library can be installed with `pip` as:


`python3 -m pip install lib_name`


## Optional if kernel name is not visualised


`python3 -m ipykernel install --user --name=pros --display-name "pros"`


## Optional for GIT and VC


If Denied permission, due to Personal Access Token (PAT) expiry and `The requested URL returned error: 403` upon `git push`


`printf "protocol=https\nhost=github.com\n" | git credential-osxkeychain erase`


Then upon `git push` one can reinsert the new credentials