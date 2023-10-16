# GPU-Accelerated Deep Learning-based Correlation Attack on Tor Networks

This correlation attack is an implementation of [DeepCoFFEA Attack](https://github.com/traffic-analysis/deepcoffea.git) but with reduced correlation time.

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
