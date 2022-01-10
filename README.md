# MoCo v2 implementation for the course Advance topics in Deep Learning (cs236605)
This repo is a Pytorch implementation of [MoCo paper](https://arxiv.org/abs/1911.05722) along with the modifications of [MoCo v2 paper](https://arxiv.org/abs/2003.04297).

## Setup
To install the requirements run:
```
pip install -r requirements.txt
```
### Download
1) Download the dataset
```
bash scripts/download.sh imagenette2
```
Or Download from: [Imagenette](https://github.com/fastai/imagenette) and organize it as follows:
```
  MoCo_implementation
  ├── datasets
  │   ├── imagenette2
  ├── saved_ckpt
  ├── main.py
  ├── Imagenette.py
  ├── linclassifier_model.py
  ├── model.py
  ├── config.yaml
```
2) Download our pre-trained model
```
bash scripts/download.sh moco
```

3) Run Test
```
./main.py --pre_trained ./saved_ckpt/
```

4) Or train from scratch 
```
./main.py
```

## Results

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">top-1 accuracy</th>
<th valign="bottom">checkpoint</th>
<!-- TABLE BODY -->

<tr><td align="left">MoCo v2 best</td>
<td align="center">143</td>
<td align="center">0.945</td>
<td align="center"><a href="https://technionmail-my.sharepoint.com/:u:/r/personal/shrout_oren_campus_technion_ac_il/Documents/236605/MoCo_v2/moco-epoch=143-val_linear-acc=0.95.ckpt?csf=1&web=1&e=gZWkJd">download</a></td>
</tr>
</tbody></table>

### Loss and Top-1 accuracy MoCo

#### Moco Loss
![MoCo_loss](/images/moco_loss.svg)

#### MoCo Accuracy
![MoCo_acc](/images/moco_acc.svg)

### Loss and Top-1 accuracy Linear model (On last MoCo Model) 

#### Linear Loss
![Lin_loss](/images/lin_loss.svg)

#### Linear Accuracy
![Lin_acc](/images/lin_acc.svg)


## Disclaimer
- This code does not support batch shuffle as in the original paper.
- The code was tested on 3 different Nvidia GPU's: 
  - GeForce RTX 2080 Ti.
  - RTX A5000.
  - Tesla V100.


## References
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```

