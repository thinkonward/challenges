# Challenge Final Submission Template

## Motivation
In the realm of data science challenges, the quality and  organization of submissions are crucial for
reproducibility, transparency, and overall effectiveness of the competition. We've observed varying degrees of
structure and documentation in submitted code, ranging from meticulously organized to somewhat chaotic.
This inconsistency poses challenges for our evaluation process and poor submission documentation
may cost you valuable points on the way to the top of the final leaderboard.  

Hence, we have developed this repository to offer you our view on a well-structured and organized final submission file
for our challenges. Our aim is to provide you with a clear framework for structuring your code and
documentation. By adhering to the guidelines, you
can streamline your workflow and ensure their solutions are easily reproducible. 

## Disclaimer
This repository serves as a general guideline and does not form any part of the terms and conditions
of any challenge. Submissions that do not adhere to these guidelines will be evaluated on a general basis, 
provided they are reproducible.


## What files are expected in a final submission?
A valid final submission file should include a specific set of files outlined on the challenge description page under 
the 'Final evaluation' subpart. Typically, the following files are required:

* Jupyter notebook (or a set of notebooks)
* Supplemental code and data
* Model checkpoints
* requirements.txt file.

## What files can be optionally included?
You may choose to include the following optional files to enhance the reproducibility and interpretability 
of your submission:

* README.md file for the submission or any of its parts
* Custom visualizations (e.g., charts, histograms, images, videos)
* Any other files that you deem appropriate to improve submission reproducibility and interpretability.

## Please refrain from attaching the following files to your submission (unless required by challenge rules):
* challenge train/test data;
* virtual environment files;
* anything unrelated to the challenge
* any personal information

## More details about required files

#### 1. Jupyter Notebook   

The Jupyter Notebook (.ipynb file) should be a high-level representation of your solution pipeline. We are not limiting
you to any specific notebook structure. For example, you can organize an entire pipeline into a notebook 
(as it's done in our sample submission file) or split it up across different notebook files (e.g. preprocessing notebook, 
training notebook, inference notebook).

In order to boost your submission interpretability score we strongly encourage you to include a detailed and informative
description of your solution in a markdown part of the notebook. Feel free to use all kinds of supplemental materials for
it inside the notebook (e.g. flow charts, screenshots, EDA plots, links to paper/documentation)


#### 2. Supplemental code and data

Feel free to organize your helping functions and classes into python modules and import them into the notebook. 
Stick to PEP 8 Style Guide for Python Code to contribute additional points to interpretability score. 
You can also wrap up entire pipeline parts into python
modules (e.g. ```preprocess.py```, ```train.py```), but each of this pipeline steps should be triggered from within the
main jupyter notebook or imported into the notebook as a module. 

If you used additional data to train a model (third-party public data), please include it in your submission 
or create a script to download it.

#### 3. Model checkpoint

Include your best model checkpoint file into submission file. Do not attach more than one checkpoint for the same model. 
If your solution pipeline involves ensemble models, please include one checkpoint for each of the models used. We encourage you to add your model checkpoints to Hugging Face. 


#### 4. Requirements.txt

Requirements.txt file must contain a list of dependencies required to reproduce the whole solution pipeline. Include 
only those libraries and packages which are used throughout your codebase. Please, refrain from using ```pip list``` 
or ```pip freeze``` commands to generate a requirements.txt. If you want to generate the requirements.txt file
automatically, you can use ```pipreqs``` library like this:

```pip install pipreqs```  
```pipreqs path/to/submission/root  --scan-notebooks```

The pipreqs library will find all dependencies being imported throughout your code and will generate a proper
```requirements.txt``` file for you. The ```--scan-notebooks``` option will ensure that dependencies for your jupyter 
notebook files will also be taken into account.


## How to get the max score for interpretability?
To attain the highest interpretability score, ensure the following criteria are met:

* Python code is styled and formatted according to PEP8 standards.
* Docstrings and inline comments are included throughout the codebase.
* Clear instructions on setting up and reproducing the solution pipeline are provided in the submission.
* A clear and informative description of the approach/idea applied is included.
* Exploratory Data Analysis (EDA) findings and insights are documented in the submission file.










