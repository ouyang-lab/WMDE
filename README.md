# WMDE
Weighted Model-based Embedding (wMDE) is a framework that combines probabilistic modeling with Euclidean distance embedding to predict 3D structures of RNA using only RNA-RNA interaction data.

# Software Pre-requisites
- R
- Matlab

# Instructions to run WMDE

## 0. Convert RNA-RNA interaction data (SHARC-seq) to contact matrix (optional)
Read the raw SHARC-seq data into R, use `Data conversion.Rmd` to generate contact matrix at certain resolution.

## 1. Input file preparation
Place contact matrix file and MATLAB scripts under a same folder and name the contact matrix file as `contact.txt`

## 2. Run the algorithm
Open `run_WMDE.m`, change to desired output directory, simply run the script.

## 3. Output files
- `predP.txt`: Predicted 3D structure
- `beta.txt`: Optimal conversion parameter
