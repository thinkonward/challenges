# Image Impeccable Final Submission

## How to Use this Submission for Prediction of Final Hold-out Test Data

1. **Install Required Packages**:  
   Run the following command in your environment to install all necessary dependencies:
   pip install -r requirements.txt
   
2. **Prepare Test Dataset:**
Extract the test dataset (or hold-out dataset) to the test_data folder.
A sample folder structure can be found in the "others note notebooks etc." section.

3. **Run Prediction Notebook:**
Open the 3_Test_Prediction.ipynb notebook and run all the cells. This will predict the test dataset and pack the results into the submission format.

4. **Locate the Submission File:**
After running the notebook, the submission file will be generated in the submission_files folder (default location).

## How to Retrain the Model from Scratch

1. (If not already done) **Install Required Packages**:  
   Run the following command in your environment to install all necessary dependencies:
   pip install -r requirements.txt

2. **Prepare Training Dataset:**
Extract the training dataset. A sample folder structure can be found in the "others note notebooks etc" folder.

3. **Preprocess the Training Data:**
   - Open and run the `1_Data_Exploration_Analysis_Preparation.ipynb` notebook.
   - This will preprocess the training dataset and store the processed data in the `training_data_processed` folder (default location).

4. **Train the Model:**
   - Open and run the `2_Training.ipynb` notebook.
   - This will train the model and generate checkpoint files in the `checkpoint` folder (default location).

5. **Predict the Test Dataset:**
   - Open and run the `3_Test_Prediction.ipynb` notebook.
   - Make sure the checkpoint name used is the same as produced by previous notebook.
   - Running all cells will predict the test dataset and package the results into the required submission format.

6. **Locate the Submission File:**
   - The submission file will be generated in the `submission_files` folder (default location).

### **Experiment Setup:**
   This experiment was executed on a device with the following specifications:
   - **Device:** Alienware x14  
   - **RAM:** 32GB  
   - **GPU:** Nvidia 3060 (6GB VRAM)
   Predicting 15 files of test dataset takes 15 minutes.
   Training model from scratch takes around 13 hours.