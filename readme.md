# Mikedev's Data science project template

## Folder structure
    ├── constants           <──── Contains many useful variables and functions
    │   ├── dataset.py     <──── Some costants regarding the dataset structure
    │   ├── model.py       <──── Your model hyperparameters
    │   ├── paths.py       <──── Costant variables contains many paths of the project 
    │   └── procs.py       <──── Useful functions in many situations
    ├── data               <──── Where all the data are saved
    │   ├── processed      <──── Data to submit to your model
    │   ├── processing     <──── Data saved during preprocessing phase
    │   └── raw            <──── Data not yet altered or modified 
    ├── models            <──── Persistent-saved models
    ├── notebooks         <──── Notebooks for EDA
    │   └── report        <──── Notebooks used as report
    ├── plots             <──── Pictures of plots
    ├── [predictions]     <──── If your task is to make predictions put here your predictions
    ├── report            <──── A report of the analisys
    ├── requirements.txt  <──── Python packages needed for this data analisys
    ├── run.py           <──── Used to run scripts
    └── src
        ├── data
        │   └── preprocess.py   <──── preprocessing data
        └── model
            └── run_model.py      <──── start the model
            
### Functions in procs.py

- [adversial validation](http://manishbarnwal.com/blog/2017/02/15/introduction_to_adversarial_validation/)
- parallel operations on [Pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) using [multiprocessing module](https://docs.python.org/3.6/library/multiprocessing.html)
- correlation matrix for categorical variables using [Cramer's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)
- save your prediction results in uniquely identifiable format using [datetime module](https://docs.python.org/3.6/library/datetime.html)


### how to use run.py and why
In order to use run.py you must execute passing in input the name of the file to execute<br>
NOTE: the file passed in input must contain a function called _main_<br>
The advantages from _run.py_ is that its checks every time that all the packages in _requirements.txt_ are satisfied
and add all the folders under src to the _PYTHONPATH_ <br>
Some examples:
`python run.py preprocess`, `python run.py run_model`