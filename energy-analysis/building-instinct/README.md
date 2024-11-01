# Building Instinct: Where power meets predictions

<img src="assets/building-instinct-image.jpg" alt="building instinct header" width="600"/>

## 🗒️ Description

This repository contains the winning submissions for the data science challenge [Building Instinct: Where power meets predictions](https://thinkonward.com/app/c/challenges/building-instinct) held by [Think Onward](https://thinkonward.com) which ran from July to October 2024. 

**The final submissions in this repository are all open source**. Use these solutions to help you electrify the discourse on energy efficiency and spark a brighter, more sustainable future for all - after all, efficiency is the ultimate victory.


## ℹ About the challenge

This was the next in the line of power focused challenges focused on power and consumption data. In this challenge participants were asked to solve a hierarchical classification problem based on building load profiles.

### 🙋 Introduction

The classification of buildings by their electricity consumption signature has significant applications. By analyzing load profiles, utilities can outsmart energy costs and identify big consumers of energy to market time-of-use tariffs and demand response programs effectively. Identifying attributes such as residential or commercial status, occupancy, and size allows utilities to target specific groups for demand response initiatives. Urban planners and policymakers use this data to design more energy-efficient cities, implement zoning regulations, and promote sustainable energy use. The insights also help real estate developers and property managers enhance energy efficiency in their buildings, reduce operational costs, and improve marketability.

### 🏗️ Challenge Structure

This was a structured challenge, which means that Challengers were asked to maximize their score on the predictive leaderboard as outlined in the evaluation section below.

### 💽 Data

The dataset provided for this challenge comprised time-stamped electricity load profiles from January 1st to December 31st for 7200 and 1440 buildings in the train and test datasets, respectively. In addition to the load profile dataset, ThinkOnward provided metadata for 7200 buildings. Buildings were classified as either residential or commercial.

These data were derived from the End-Use Load Profiles for the U.S. Building Stock data created by the national Renewable Energy Laboratory which are licensed under the CC BY 4.0 license [link](https://creativecommons.org/licenses/by/4.0/) and can be referenced at [https://dx.doi.org/10.25984/1876417](https://dx.doi.org/10.25984/1876417).

### 📏 Evaluation

A customized hierarchical F1-score measured the performance of the classification models. The final F1-score (the predictive leaderboard score) was derived by first calculating the F1-scores at two hierarchical levels, and combining them into a weighted average. For technical details, check out the [Challenge page](https://thinkonward.com/app/c/challenges/building-instinct) and Starter Notebook in this repository.

### 👏 Knowledge Sharing
In keeping with our goal of collaboration and knowledge sharing, the winners solutions for this challenge are available in this directory for you to learn from and grow as a data scientist in the energy space. Remember to include license files and acknowledgements as part of the open-source community. 