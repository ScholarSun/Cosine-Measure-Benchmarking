# Cosine Measure Problem - Benchmarking and Test Set
![GitHub](https://img.shields.io/badge/License-GPL%20v3-blue.svg)

This repository contains the implementation and benchmarking of algorithms designed to solve the cosine measure problem. Also contained is the code used to generate a collection of sets with known cosine measure, which we use to benchmark the algorithms. 

For a detailed explanation, please see: W. Hare, and S. Sun, On the computation of the cosine measure in high dimensions, https://arxiv.org/abs/2506.19723 (2025).


## Generating the Test Set
The file `data_generators/set_generators` contains the functions used to generate a single instance of a particular type of set. The script `data_generators/test_collection_generator` generates the entire test-set. Below is a sample function call to generate the test set.
```
generate_test_set('output_directory_path', [2,3,4], 42)
```
#### Parameters
 - output_path : string : path to destination directory
 - dimensions : list[int] : specifies the dimension of sets to be generated
 - seed : int : (optional) 

## Benchmarking Algorithms
The directory `algorithms` contains the implentation of all methods tested. The script `benchmark_final.py` contains the code that facilitated the testing and the file `benchmark_results.csv` is the resulting output. 


## Contact
Please contact us via email to report any issues:
```
scholarsun99@gmail.com
```
