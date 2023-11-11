## GPU-Accelerated Deep Learning-Based Correlation Attack on Tor Networks

This is the official implementation of the paper [GPU-Accelerated Deep Learning-Based Correlation Attack on Tor Networks](https://doi.org/10.1109/ACCESS.2023.3330208)

## Abstract
```
The Tor network, renowned for its provision of online privacy and anonymity, faces the constant threat of correlation attacks that aim to compromise user identities.
For almost two decades, these correlation attacks were based on statistical methods. However, in recent years, deep learning-based correlation attacks have been introduced to make them more accurate.
Nevertheless, in addition to being accurate, fast correlation attacks on Tor are crucial for assessing the real-world viability of such attacks because reduced correlation time aids in estimating its practical implications.
Moreover, a reduction in correlation time also helps improve efficiency and ensures practical relevance of the attack.
The existing state-of-the-art implementation of a correlation attack on Tor suffers from slow performance and large memory requirements.
For instance, training the model required 133 GB of memory, and correlating 10,000 flows takes about 976 seconds.
In this paper, we present a novel GPU-based correlation strategy and a fast traffic flow loading technique to reduce time complexity by 7.12x compared to existing methods.
Our computational approach, reliant on PyCUDA, facilitates the parallelization of operations used in the attack, thereby enabling efficient execution through the utilization of GPU architecture.
Leveraging these two approaches, we introduced an improved correlation attack, which shows high accuracy and fast performance compared to state-of-the-art methods.
Moreover, we address resource limitation issues by reducing memory consumption by 47.37% during the training phase, which allows the model to be trained with much fewer resources.
```

## How to run the code?

0. Run the following command at the terminal to install the required libraries:
```
pip install -r requirements.txt
```

1. - Download the training data from [here](https://drive.google.com/drive/folders/1PG0sF6AHHn_2LxyoIztwjpoxDmB7r39z?usp=sharing) and move it into the `./data/datasets/train_data` folder.
   - Download the 10k testing data from [here](https://drive.google.com/drive/folders/1JUC-KBghWX42yg19gYDcrospyuE16d6X?usp=sharing) and move it into the `./data/datasets/test_data` folder.
   - Run `divide_10k_test.ipynb` to divide the 10k testing data into separate 2094, 5000, 7500 and 10,000 testing flows.
  
2. Run `train_fens.py` file to train the deep learning model for correlating the Tor and exit flows.

3. Run `eval_dcf_pycuda.py` file for evaluation of correlation time on the created testing data i.e., 2094, 5000, 7500, and 10,000 flows.



## Citation

If you think our code is useful, please cite our paper:
```
@ARTICLE{10309127,
  author={Hafeez, Muhammad Asfand and Ali, Yasir and Han, Kyung Hyun and Hwang, Seong Oun},
  journal={IEEE Access}, 
  title={GPU-Accelerated Deep Learning-Based Correlation Attack on Tor Networks}, 
  year={2023},
  volume={11},
  number={},
  pages={124139-124149},
  doi={10.1109/ACCESS.2023.3330208}
}
```


## Acknowledgement

We appreciate the authors of DeepCoFFEA for their valuable work on correlation attack on Tor.

Here is the link to their GitHub Repo.
https://github.com/traffic-analysis/deepcoffea