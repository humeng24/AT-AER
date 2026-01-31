
# AT-AER: Adversarial Training with Adaptive Example Reuse (submitted CAAI Transactions on Intelligence Technology)



This repository is a PyTorch implementation of the AT-AER. The paper has been submitted to CAAI Transactions on Intelligence Technology.


## Framework
The framework of Margin-SNN.

<img src="figs/framework.png" width="500"/>

## High-ordered AE

<table?
  <tr>
     <td> <center> <img src="figs/a0c10202-5df0-463e-b191-58a0ae6db95c.png" width=500/> (a) Progressive AE evolution of CIFAR-10 </center> </td>
     <td> <center> <img src="figs/f7e47d16-5986-4aff-824c-badb5d4c616d.png" width=500/> (a) Progressive AE evolution of CIFAR-100 </center> </td>
     <td> <center> <img src="figs/ffd6437e-a8b5-48b0-85c1-bd3ed91e7042.png" width=500/> (a) Progressive AE evolution of SVHN </center> </td>
   </tr>
</table>

#### Progressive AE evolution of CIFAR-10


#### Progressive AE evolution of CIFAR-100
<img src="figs/f7e47d16-5986-4aff-824c-badb5d4c616d.png" width=500/>

#### Progressive AE evolution of SVHN
<img src="figs/ffd6437e-a8b5-48b0-85c1-bd3ed91e7042.png" width=500/>

## Ablation of AT-AER 

#### Training process of AT-AER⌝R
<img src="figs/8ec7dffc-877d-45c0-b18b-f306ce7807ab.png" width=500/>
#### Training process of AT-AER⌝L
<img src="figs/c82788eb-7f90-4f39-ba8d-e54058f137ac.png" width=500/>
#### Training process of AT-AER⌝W
<img src="figs/30d48614-2434-49f0-9758-6b3bd021bbd5.png" width=500/>
#### Training process of AT-AER⌝A
<img src="figs/fb86161f-18a9-47bc-b766-8eaf1e8231bd.png" width=500/>
#### Training process of AT-AER
<img src="figs/02cdfcf5-ab14-40c0-8db6-cfff11cf5241.png" width=500/>

## train AT-AER
 
``nohup python main.py --cuda 0 --dataset cifar10 --savemodels ./savemodels-cifar10 --logs ./logs-cifar10 > train_cifar10.log 2>&1``

AT-AER can be trained on other datasets in the same way as cifiar10



## Citation
```
@article{Hu2026AT_AER,
  author = "Meng, Hu and Yanting, Guo and Ran, Wang and Xizhao, Wang, Rihao, Li and Qin, Wang",
  title = "AT-AER: Adversarial Training with Adaptive Example Reuse",
  journal = "submitted to CAAI Transactions on Intelligence Technology",
  year = "2026"
}
```





