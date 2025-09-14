# WMDE
Weighted Model-based Embedding (wMDE) is a framework that combines probabilistic modeling with Euclidean distance embedding to construct 3D structures.

# Software Pre-requisites
- R
- Matlab

# Instructions to run WMDE

## 1. Input file preparation
Place contact matrix file and MATLAB scripts under a same folder and name the contact matrix file as `contact.txt`

## 2. Run the algorithm
Open `run_WMDE.m`, change to desired output directory, simply run the script.

## 3. Output files
- `predP.txt`: Constructed 3D structure
- `beta.txt`: Optimal conversion parameter
