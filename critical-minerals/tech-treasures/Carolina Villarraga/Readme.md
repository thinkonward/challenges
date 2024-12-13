# Tech Treasures Challenge - Critical Minerals Exploration Dashboard

## Overview
This repository is part of the **[Tech Treasures Challenge](https://getech.com/news/getech-collaborates-with-thinkonward-to-develop-ai-powered-critical-minerals-exploration-dashboard/?utm_campaign=2024%20Challenges%20and%20Bounties&utm_content=304316993&utm_medium=social&utm_source=linkedin&hss_channel=lcp-40889185)**. The goal is to develop an AI-powered dashboard for critical minerals exploration.

Throughout this project, several approaches were explored in the quest to build a model for predicting rare earth element (REE) deposits in British Columbia. Initial ideas focused on incorporating raster data, linear features, and various geochemical and geological attributes into ensemble models. A PCA was conducted to reduce dimensionality, and unsupervised models like K-means were applied. Unfortunately, these models did not yield significant clustering results, likely due to the complexity of the data or insufficient training samples.

A Random Forest model was later implemented, but it performed poorly, potentially due to the added complexity of multiple categories in the dataset. The analysis centered on 65 points of Critical Mineral showings, divided into mineralized and non-mineralized categories with varying confidence levels. While this seemed to provide a balanced dataset, it may have introduced more complexity than expected. Going forward, the model will be refined outside of the challenge, reducing the categories to two and utilizing the `grid_no_bc` mesh to simplify the process.

## Repository Structure
- **`datasets/`**: Contains all relevant datasets, including:
  - **raster**:
   - **originals/**: The original raster files used in the project. These are kept in the repository for size management and proper functioning of the analysis and dashboard. (at the end due constraints of ZIP.file submissions only the ones that are utilized for random forest were kept in the folder)
    - **resampled/**: Contains rasters that have been resampled to a standardized resolution. These files are used in the final analysis.
    - **cropped_reprojected/**: Rasters that have been both cropped and reprojected to the required CRS. These rasters are generated through preprocessing in the first notebook and are computationally expensive. They are not included in the repository.
    - **float/**: Rasters converted to a floating-point format for further analysis. These files are also generated through preprocessing and are not included in the repository.
 - **shapefiles**
- **`documents/`**: Reference materials and auxiliary documents.
- **`models/`**: Any saved models used during the analysis process.
- **`scripts/`**: Python scripts for data preprocessing and utility functions.

### Notebooks:
1. **`1_Data_Collection_Cleaning.ipynb`**: Data cleaning and transformation steps.
2. **`2_Tech_Treassure_for_REE_minerals.ipynb`**: Main notebook with the analysis pipeline for REE exploration.
3. **`3_legacy.ipynb`**: Contains earlier versions and approaches that were explored but not fully implemented.

### Dashboard:
- **`dash.py`**: This file contains the Streamlit dashboard code. It visualizes the critical mineral exploration data and allows users to interact with geological layers and view predictions.
  - To run the dashboard locally:
    ```bash
    streamlit run dash.py
    ```
  - You can also access the deployed version of the dashboard [here](#) (add the deployment link).

    ```
### Map Outputs:
- **`predicted_prospectivity_map_discrete.tif`**: The resulting prospectivity map generated from the analysis.
---
## Next steps
- **Incorporate More Data**: Adding geochemical data and other geological layers could refine predictions.
- **Synthetic Data**: Potential to generate synthetic non-mineralized samples to improve model balance.
- **Model Tuning**: Further hyperparameter tuning to improve classification accuracy.

## Contact
[RETRACTED]

---