# PhaseGAN: A deep-learning phase-retrieval approach for unpaired datasets
PhaseGAN is a deep-learning phase-retrieval approach allowing the use of unpaired datasets and includes the physics of image formation. 
For more detailed information about PhaseGAN training and performance, please refer to the [PhaseGAN paper](https://arxiv.org/abs/2011.08660).

## Colab Notebook
PhaseGAN Tutorial [Google Colab](https://colab.research.google.com/github/pvilla/PhaseGAN/blob/master/PhaseGAN_Notebook.ipynb) | [Code](https://github.com/pvilla/PhaseGAN/blob/master/PhaseGAN_Notebook.ipynb)

## Getting Started
### Prerequisites

- Linux (not tested for MacOS or Windows)
- Python3
- NVIDIA GPU (not tested for CPU)

### Installation

Clone this repo:

```
git clone https://github.com/pvilla/PhaseGAN.git
cd PhaseGAN
```
To install the required python packages:

```
pip install -r requirements.txt
```

## Training
For the training, our provided data loader [Dataset2channel.py](https://github.com/pvilla/PhaseGAN/blob/master/dataset/Dataset2channel.py) support loading data with HDF5 format. An example of the dataset structure could be find in [Example dataset folder](https://github.com/pvilla/PhaseGAN/tree/master/dataset/example_dataset) and [PhaseGAN validation dataset (Google Drive)](https://drive.google.com/drive/folders/1rKTZYJa54WeG-2TikoXpdRcqTiSQ-Ps5?usp=sharing).
Please note that ground truth images are provided for validation purposes, but we never use it as references in the unpaired training. 

We used hdf5 data format for the original training. For the training with other data formats, you may want to create a customized data loader. 

To run the training:

`python3 train.py`

For more training options, please check out:

## run with initial epoch 0
```
python3 train.py --load_path dataset/IMGS_PCI -b 10 --model_A UNet --model_B UNet
```

## run with initial epoch # (continue training)
```
python3 train.py --load_path dataset/IMGS_PCI --load_weights True --model_A UNet --model_B UNet
```
input the initial epoch: 3
input the corresponding date (ex: Jun04_19_31): May04_00_39
```
python3 train.py --load_path dataset/IMGS_PCI --load_weights True --model_A WNet --model_B SRResNet
```
input the initial epoch: 5*
input the corresponding date (ex: Jun04_19_31): Jun04_19_31*

python3 train.py --load_path dataset/IMGS_PCI --load_weights True --model_A UNet --model_B UNet
input the initial epoch: 4*
input the corresponding date (ex: Jun04_19_31): Jun04_19_31*

## Testing
```
python3 test.py --load_path dataset/IMGS_PCI -b 1 --model_A UNet --model_B UNet
```

## Results
The training results will be saved in: `./results/fig/run_name/train`.
The training parameters and losses will be saved to a txt file here: `./results/fig/run_name/log.txt`.
The models will be saved in `./results/fig/run_name/save`.

## visualization results (training/validation curves) 
```
tensorboard --logdir './runs/.' --bind_all
```

## Citation
If you use this code for your research, please cite our paper.
```
@article{zhang2020phasegan,
  title={PhaseGAN: A deep-learning phase-retrieval approach for unpaired datasets},
  author={Zhang, Yuhe and Noack, Mike Andreas and Vagovic, Patrik and Fezzaa, Kamel and Garcia-Moreno, Francisco and Ritschel, Tobias and Villanueva-Perez, Pablo},
  journal={arXiv preprint arXiv:2011.08660},
  year={2020}
}
```
## Acknowledgments
Our code is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [TernausNet](https://github.com/ternaus/TernausNet).
