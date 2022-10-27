
# Contextual-assisted Scratched Photo Restoration (TCSVT 2022)


**Paper**: 



> **Abstract:** *AbstractPrinted photographs can be easily warped, wrinkled,
and even deteriorated over time. Existing methods treat the
restoration of scratches as a pure inpainting problem that
neglects the underlying corrupted contextual knowledge. They
totally remove the scratched texture and fill in the missing
holes according to the background. Obviously, they discard very
insignificant semantic contextual information. In this paper, we
propose an automatic retouching approach for the scratched
photograph with the aids of scratch/background contexts. We
explicitly process scratch and background contexts in two stages.
In the first stage, we mainly extract global scratch features, while
the mask is introduced in the second stage to filter out and
inpaint the scratches. Both contexts are carefully reciprocated
for a faithful restoration. Particularly, we propose a Scratch
Contextual Assisted Module (SCAM) to adaptively learn texture
within the detected mask. This module utilizes the distance
between the scratch mask-out feature and scratch encoder feature
for modeling the pixel-wise correspondence, which determines
the importance of the encoder feature within the scratch mask.
Furthermore, to facilitate the evaluation of scratch restoration
methods, we create two new scratched photo datasets which have
238 scratch/scratch-free photo pairs to promote the development
in the scratch restoration field, namely Old Scratched Photo
Dataset (OSPD) and Modern Scratched Photo Dataset (MSPD).
Extensive experimental results on the proposed datasets demon-
strate that our model outperforms existing methods.* 



## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Quick Run

## Training and Evaluation

## Training

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

- Test the model
```
python test.py
```

## Citation

    @inproceedings{Zamir2021MPRNet,
        title={Multi-Stage Progressive Image Restoration},
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
                and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
        booktitle={CVPR},
        year={2021}
    }
    
    @inproceedings{Liu2019MEDFE,
  title={Rethinking Image Inpainting via a Mutual Encoder-Decoder with Feature Equalizations},
  author={Hongyu Liu, Bin Jiang, Yibing Song, Wei Huang, and Chao Yang,},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}

## Contact
Should you have any question, please contact cweiwei349@gmail.com
