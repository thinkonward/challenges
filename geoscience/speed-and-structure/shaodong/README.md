### Challenge: Speed and Structure

https://thinkonward.com/app/c/challenges/speed-and-structure


### Team: [REDACTED]
### Email: [REDACTED]

### Python Version: 3.10.8

### Project Structure 

* ```~/solution_pipeline.ipynb``` - main notebook file, pipeline entrypoint;
* ```~/src``` - source code directory with helping modules for the pipeline and the software License;
* ```~/images``` - images for notebook markdown;
* ```~/data``` - placeholder directory for the pipeline data;
* ```~/experiments``` - placeholder directory for save the training result;
* ```~/train_txt``` - directory with the txt file after the training data is divided into five folds, and I use f0 to train my models.
* ```~/libs``` - directory contains required library files.
* ```~/install.sh``` - scripts for installing the required environment.
* ```~/my_checkpoints``` - directory with my best model weights. All models were trained on f0 data. 10 custom_eva02_base_split_at_6 models were trained using different random seeds. And the ensemble of these 10 models resulted in a cv score of 0.020559 and a public lb score of 0.023526. 6 custom_base_eva02_split_at_8 models were trained using different random seeds. And the ensemble of these 6 models resulted in a cv score of 0.020876 and a public lb score of 0.023591. 6 custom_tiny_eva02_split_at_8 models were trained using different random seeds. And the ensemble of these 6 models resulted in a cv score of 0.020851 (public lb scores were not tested). <font color=red>The ensemble of all 22 models above resulted in a cv score of 0.02054 and a public lb score of 0.023391, which is my highest public lb score</font>
  
|   models     |                   local cv score	  |   public lb score  |
|  ----  | ----  | ----  |
| custom_eva02_base_split_at_6 (10 seed ensmble) |  0.020559	   |0.023526         |
| custom_eva02_base_split_at_8 (6 seed ensemble) |  0.020876	   |0.023591         |
| custom_eva02_tiny_split_at_8 (6 seed ensemble) |  0.020851	   |? (did not test) |
| ensemble all the above models	                  |  0.02054	   |0.023391         |



### Solution Reproduction Note
to  reproduce the solution, follow the instructions in ```solution_pipeline.ipynb```
