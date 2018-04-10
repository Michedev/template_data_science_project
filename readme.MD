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

- adversial validation
- parallelism map in dataframe
- correlation matrix for categorical variables
- save your prediction results in uniquely identifiable format


### how to use run.py and why
In order to use run.py you must execute passing in input the name of the file to execute
NOTE: the file passed in input must contain a function called 'main'<br>
The advantage from run.py is that its checks every time that all the requirements are satisfied
and add all the folders under src to the PYTHONPATH<br>
Some examples :
`python run.py preprocess`, `python run.py run_model`