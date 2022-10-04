# MRF-UNets: Searching UNet with Markov Random Fields

## Prerequisite
### Dependency
* The scripts depend on the following packages

```
cv2
PIL
thop
numpy
torch
pgmpy 
albumentations
osgeo
pydicom
SimpleITK 
```

* Some packages are difficult to install and they are only used in data preprocessing, e.g. `osgeo`. You do not have to install all packages if you are not interested in some datasets. Please refer to `datas/preprocess.py` and comment out the related code lines.

### Data Preparation
* Preprocess a dataset

```Shell
python datas/preprocess.py func data_dir
```

For example

```Shell
python datas/preprocess.py Land "/Users/whoami/datasets"
```

* The data hierachy before and after the preprocessing should be as follows. Please refer to `datas/preprocess.py` for more details.

```
data_dir
|— land
|  |- train
|  |- resized
|- road
|  |- train
|  |- resized
|- building
|  |- spacenet
|  |  |- AOI_2_Vegas_Train
|  |  |- AOI_3_Paris_Train
|  |  |- AOI_4_Shanghai_Train
|  |  |- AOI_5_Khartoum_Train
|  |- train
|  |- resized
|- chaos
|  |- train
|  |  |- CT
|  |  |- MR
|  |- resized
|- promise
|  |- train
|  |  |- TrainingData_Part1
|  |  |- TrainingData_Part2
|  |  |- TrainingData_Part3
|  |- resized
```

## Usage
### Learning
* Learn a MRF

```Shell 
python search.py
```

### Inference
* Inference over the learnt MRF

```Shell
# diverse 5-best inference
python inference.py --m 5 --lam 10
# diverse 10-best inference
python inference.py --m 10 --lam 20
```

### Training
* Train a found architecture

```Shell
# MRF-UNetV1
python train.py --choices "8,9,2,4,0,4,8,6,2,1,8,3,3,3,0,7,5,1,8,2,0,3,0,1,4,0"
# MRF-UNetV2
python train.py --choices "8,8,3,3,1,3,3,1,3,3,1,3,3,1,0,8,1,0,8,1,0,8,1,0,8,1"
```

* If you just want to benchmark with MRF-UNets, copy `models/mrf_unet.py` and `models/ops.py` into your codebase and add the following statements into your training script

```Python
from models.mrf_unet import ChildNet
model = ChildNet(image_channels, num_classes, channel_step, choices)
```

## Citation
```BibTeX
@InProceedings{Wang2022MRF-UNets,
  title     = {MRF-UNets: Searching UNet with Markov Random Fields},
  author    = {Wang, Zifu and Blaschko, Matthew B.},
  booktitle = {ECML-PKDD},
  year      = {2022}
}
```