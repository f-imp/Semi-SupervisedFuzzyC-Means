# Semi-SupervisedFuzzyC-Means

This is an implementation of Semi-Supervised Fuzzy C-Means (SSFCM) algorithm.
It has been developed three different version of the SSFCM algorithm:
* For-loop
* Matrix multiplication
* Matrix multiplication with GPU accellerator

Both the quality of the clustering results and the running time execution has being compared.

# Built With
* Python 3.9


# Getting Started
## Prerequisites
These are the libraries/packages that must be install within your python environment:
| Package | Min-Version |
| :----- | :- |
| Numpy |1.22.3|
| Scipy | 1.8|
| Cupy | 10.3.1|
| Matplotlib| 3.5.1|
| Pandas | 1.4.2|
| Seaborn | 0.11.2|
| Shutil | - |


## Usage
To perform the experiments locally it is necessary to have a GPU (with CUDA installed) or it could be used a Colab runtime with the hardware accelleration 

### Setup 
It has been used different parameters to perform the experiments with a supervision of 20% of the total number of samples:
| Parameters | Value |
| :----- | :- |
|#samples |100, 1000, 10000|
|#features|10, 100, 500|
|#clusters| 3, 5, 10|

### Local Execution
```sh
$ python Semi-SupervisedFuzzyC-Means/main.py
```
### Google Colab

Using the notebook file available at the following link:

```sh
https://drive.google.com/file/d/1rsKU-c-zb3_-M8cBNJMpbU9Qh_NAuTaQ/view?usp=sharing
```
## Results
### Comparison among all 3 algorithms
![Bar Chart Execution Time_all](Result/output.png)
### Comparison among matrix multiplication version with and without GPU
![Bar Chart Execution Time_just matrix_versions](Result/output2.png)
### Running time matrix multiplication version with GPU
![Bar Chart Execution Time_just matrix_versions](Result/output3.png)

### Quality Assessment
![original blobs](Result/10000_10_5/Original_10000_10_5.png)
![for-loop version](Result/10000_10_5/SSFCM_v1_10000_10_5.png)
![matrix multiplication version](Result/10000_10_5/SSFCM_v2_10000_10_5.png)
![matrix multiplication with GPU version](Result/10000_10_5/SSFCM_v3_10000_10_5.png)
