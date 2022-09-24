# Differentially Private Covariance Revisited
This repo contains the official code for the implementations of the algorithms and experiments in our NeurIPS 2022 paper "Differentially Private Covariance Revisited". Paper is avaliable [here](https://arxiv.org/abs/2205.14324?context=cs).

| Folder          | Description                                                                                                |
| ----------------| ---------------------------------------------------------------------------------------------------------- |
| adaptive        | implementations for our algorithms and the Gaussian mechanism                                              |
| exponential     | implementation for [Exponential mechanism for covariance estimation](https://papers.nips.cc/paper/2019/hash/4158f6d19559955bae372bb00f6204e4-Abstract.html)                                                           |
| coinpress       | a copy of the code from https://github.com/twistedcubic/coin-press, containing implementation of [coinpress](https://proceedings.neurips.cc/paper/2020/hash/a684eceee76fc522773286a895bc8436-Abstract.html) |


## Requirements
The algorithms and experiments were implemented in Python (v3.9).

The following packages are required and can be installed via conda or pip:<br/>
pytorch v1.9.0 <br/>
numpy v1.22.3 <br/>
scipy v1.8.0 <br/>
scikit-learn v1.0.1 <br/>

```setup
conda install numpy==1.22.3
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
conda install scipy==1.8.0
conda install scikit-learn==1.0.1
```


## Evaluation
To reproduce the results in a specific figure, run its corresponding script listed below.
| File name          | Corresponding figure | Description                                           |
| ------------------ |--------------------- | ----------------------------------------------------- |
| test_d_N1.py       | Figure 1.            | Synthetic dataset test, fixing tr=1         |
| test_Ns.py         | Figure 2(a).            | Synthetic dataset test, various tr        |
| test_d.py          | Figure 2(b).            | Synthetic dataset test, various d       |
| test_n.py          | Figure 2(c).            | Synthetic dataset test, various n      |
| test_rho.py        | Figure 2(d).            | Synthetic dataset test, various rho       |
| test_mnist_fix_n.py  | Figure 3.            | MNIST dataset test        |
| test_news_rho_l2.py  | Figure 4(a)-(c).            | News dataset test, unit l2 norm        |
| test_news_rho_maxl2.py  | Figure 4(d)-(e).            | News dataset test, normalized by max l2 norm       |
| test_mnist_digit.py  | Figure 5.            | MNIST dataset individual digits test        |
| test_d_N1_pure.py       | Figure 6.            | Synthetic dataset test, fixing tr=1 (pure DP)        |
| test_Ns_pure.py         | Figure 7(a), 7(d)            | Synthetic dataset test, various tr (pure DP)      |
| test_d_pure.py         | Figure 7(b), 7(e)            | Synthetic dataset test, various d (pure DP)      |
| test_eps_pure.py         | Figure 7(c), 7(f)            | Synthetic dataset test, various epsilon (pure DP)      |

For example, to run the test in Figure 1:
```test
python test_d_N1.py
```
Note: when running the corresponding scripts for the tests on real-world datasets, the code will check whether the datasets are in the directory "./data/". <br/> If not, it will download them using the links embedded in the code to that directory. 

## Compute Resources
All experiments reported in the paper were run on a linux machine with 48 CPUs and 256GB RAM. <br/>
