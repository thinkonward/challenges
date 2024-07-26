# Power Patterns: Harnessing Electricity for Innovation
This is our submission to the [Onward Power Patterns: Harnessing Electricity for Innovation competition](https://thinkonward.com/app/c/challenges/power-patterns).   

* **Author:** [Redacted]
* **Expected Run Time:** 30s
* **Minimum Hadware Requirements:** 8 core CPU, 16 GB of RAM
* **Submission Summary:** By extracting size, temporal, and uncertainty features, we segment buildings into distinct clusters and develop targeted demand response programs. The proposed DR programs aim to improve grid stability and efficiency by incorporating measures like aggregation, seasonal adjustments, time-of-use pricing, and battery storage.

## Project Overview
Understanding how and when customers use electricity is crucial for utilities to manage the electric grid efficiently, reduce peak demand, and promote energy conservation. This project aims to:

1. Cluster Load Profiles: Employ unsupervised machine learning algorithms to group similar load profiles, revealing distinct customer segments based on their energy usage behaviors.
1. Design Demand Response Programs: Develop tailored demand response programs for each customer segment, encouraging energy efficiency and reducing strain on the grid during peak hours.

## Key Features
* Data Visualization: Employs informative visualizations to illustrate load profile patterns, clustering results, and DR program impacts.
* Feature Engineering: Extracts relevant features from load profiles to enhance clustering accuracy and interpretability.
* Unsupervised Learning: Utilizes clustering algorithms to discover natural groupings within load profile data.
* Demand Response Strategies: Recommends specific DR programs (e.g., time-of-use tariffs, peak pricing) for each customer segment to incentivize behavioral changes.

## Repository Structure
* data/: Contains sample load profile data (replace with the competition dataset).
* notebooks/: Jupyter Notebooks detailing the data analysis, clustering, and DR program design process.
* requirements.txt: List of Python packages required to run the project.

## Getting Started
1. Download and save data in `./data`.
2. Create a Virtual Environment (Recommended):
```
python -m venv venv 
```
3. Activate the Virtual Environment:
```
source venv/bin/activate
```
4. Install Dependencies:
```
pip install -r requirements.txt
```
5. Run notebook `notebooks/Power Patterns Submission.ipynb`:
```
jupyter notebook
```

