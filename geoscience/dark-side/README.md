# Dark side of the volume

<img src="assets/dark-side.png" alt="challenge header image" width="600"/>

## 🗒️ Description

This repository contains the winning submissions for the [Dark side of the volume Data Science Challenge](https://thinkonward.com/app/c/challenges/dark-side) held by [Think Onward](https://thinkonward.com) which ran from October 2024 to January 2025. 

**The final submissions in this repository are all open source**. This should help inspire you to build on this work, amplify the impact of it by sharing your solutions with the global community, and encourage peer review and collaboration. You can find the winning models on the `dark-side` [branch of the ThinkOnward HuggingFace Challenges model page](https://huggingface.co/thinkonward/challenges/tree/dark-side) 🤗.


## ℹ About the challenge

### 🙋 Introduction

The goal of this challenge was to identify faults on seismic volumes and then build a polygon around them in three dimensions. ThinkOnward provided the data for this challenge in a typical data science challenge format, but you participants were not required to use machine learning to solve the problem.  


### 🏗️ Challenge Structure

In the subsurface, faults can either be active and move, or they can be inactive and be records of past movement of the rocks. 
To image these faults we use seismic sound waves propagated into the earth and record their response as they reflect back to the surface as part of 2D and 3D seismic surveys. 
Faults appear different from the surrounding rocks. They usually are low amplitude, and generally offset the surrounding layers of rock. 
Some faults are very easy to identify, while others require additional seismic processing time to find and map. One useful process is called spectral decomposition.
Think of spectral decomposition like a prism, where you shine white light on one side, and the light that comes out the other side is split into its respective wavelengths. 
For seismic data, spectral decomposition is usually performed on the frequency domain, and you can think of it as a measure of the seismic amplitude for a given frequency band. 
The results are traditionally visualized by blending any color you like to the different frequencies. 

For this challenge, participants were asked you to find and map faults in 3D seismic volumes.

<img src="assets/dark-side-overview-1.png" alt="seismic timeslice data image" width="600"/>


### 💽 Data

Participants were provided with 400 paired synthetic seismic datasets with associated binary fault labels. The synthetic data were delivered as Numpy arrays with a shape of (300,300,1259).    

The Dark side of the volume Data by Think Onward are licensed under the CC BY 4.0 license (link)

### 📏 Evaluation

For this challenge we used a Dice Coefficient similar to the one we used for the Every Layer challenge. However, for this Challenge we used a 3D Dice Coefficient that takes predicted fault masks and compares them to the ground truth fault masks.

### 👏 Knowledge Sharing
In keeping with our goal of collaboration and knowledge sharing, the winners solutions for this challenge are available in this directory for you to learn from and grow as a data scientist in the energy space. Remember to include license files and acknowledgements as part of the open-source community. 

