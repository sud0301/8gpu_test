# 8gpu_test
cityscapes_training

## Dataset preparation

### Cityscapes
Download the dataset from the Cityscapes dataset server([Link](https://www.cityscapes-dataset.com/)). Download the files named 'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip' and extract in ```./data/cityscapes/```

## Create the environment
```
conda create -n 8gpu_test python=3.9
```
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy
```

## Training on Cityscapes Dataset on 8-GPUs
```
python train_full.py    --batch-size 40
```

Reduce batch-size, if GPU memory is not enough. 

