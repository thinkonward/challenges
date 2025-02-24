### Challenge: Dark side of the volume

### Python Version: 3.10.8

### Project Structure 

* ```~/solution_pipeline.ipynb``` - main notebook file, pipeline entrypoint;
* ```~/src``` - source code directory with helping modules for the pipeline and the software License;
* ```~/data``` - placeholder directory for the pipeline data;
* ```~/experiments``` - placeholder directory for save the training result;
* ```~/train_txt``` - directory with the txt file after the training data is divided into five folds, and I use f0 and f1 to train my models.
* ```~/my_checkpoints``` - directory with my best model weights, which was trained locally by me and got the highest score in the public lb.In detail, the model trained with f0 data scored 0.9415 on public lb, and the model trained with f1 data scored 0.9445 on public lb. The ensemble of the two models scored 0.946128.  It should be noted that my score on public lb is 0.946467, which is obtained by ensemble the effv2s_f0_cv09017_lb09415.pth model and the effv2s_f1_iter316k_cv08822_lb0945.pth model. <font color=red>But in local cv score, the ensemble of these two models is lower than ensemble of effv2s_f0_cv09017_lb09415.pth and effv2s_f1_cv08901_lb09445.pth, So I choose the latter ensemble as my final submission result, which scored 0.946128 in public lb</font>. If you want to reproduce my highest score(0.946467) on public lb, you can use the former ensemble.

|  models   | public lb score  |
|  ----  | ----  |
| effv2s_f0_cv09017_lb09415.pth | 0.9415 |
| effv2s_f1_cv08901_lb09445.pth | 0.9445 |
| effv2s_f1_iter316k_cv08822_lb0945.pth | 0.9452 |
| effv2s_f0_cv09017_lb09415.pth + effv2s_f1_cv08901_lb09445.pth | 0.946128 |
| effv2s_f0_cv09017_lb09415.pth + effv2s_f1_iter316k_cv08822_lb0945.pth  | 0.946467 |
* ```~/pretrained_model``` - directory with tf_efficientnetv2_s timm official ImageNet21K dataset pretrained weights, which is used for pretrained weights when training;
* ```~/images``` - images for notebook markdown;
* ```~requirements.txt``` - list of required dependencies;

### Solution Reproduction Note
to  reproduce the solution, follow the instructions in ```solution_pipeline.ipynb```
