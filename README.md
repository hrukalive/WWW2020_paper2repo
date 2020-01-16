# WWW2020_p2r
Install required dependencies

1. Follow the instructions on [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/) to install pyTorch 1.2.0
2. Install pytorch-lightning 0.5.0
```
$ pip install pytorch-lightning==0.5.0
```
3. Install pytorch_geometric (make sure Cuda is correctly installed)
```
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

Download the dataset of the chosen size into the `src/data` folder, go to `src/` and then preprocess the data with
```
$ python data_maker.py
```

Finally, start training with
```
$ python train.py --exp_name <experiment name>
```

Note: In order to use GPU for training, we suggest you install Apex by
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
In the case of training on CPU or not wanting to use Apex, please comment out the line 45 and 46 in `train.py`.

