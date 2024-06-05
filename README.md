# ReLUs Are Sufficient for Learning Implicit Neural Representations

Code to reproduce experiments in (https://arxiv.org/abs/2406.02529)

Abstract: Motivated by the growing theoretical understanding of neural networks that employ the Rectified Linear Unit (ReLU) as their activation function, we revisit the use of ReLU activation functions for learning implicit neural representations (INRs). Inspired by second order B-spline wavelets, we incorporate a set of simple constraints to the ReLU neurons in each layer of a deep neural network (DNN) to remedy the spectral bias. This in turn enables its use for various INR tasks. Empirically, we demonstrate that, contrary to popular belief, one can learn state-of-the-art INRs based on a DNN composed of only ReLU neurons. Next, by leveraging recent theoretical works which characterize the kinds of functions ReLU neural networks learn, we provide a way to quantify the regularity of the learned function. This offers a principled approach to selecting the hyperparameters in INR architectures. We substantiate our claims through experiments in signal representation, super resolution, and computed tomography, demonstrating the versatility and effectiveness of our method. The code for all experiments can be found at (https://github.com/joeshenouda/relu-inrs).

To get started right away you can check out the following [Google Colab](https://colab.research.google.com/drive/1LQbGQTBodIhtgiqJLsttet5EFq9oWi84?usp=sharing)

## Setup

With conda:

```
conda create -n relu-inrs python=3.9
conda activate relu-inrs
pip install -r requirements.txt
```

Then download the data by running the following in your terminal:

```gdown --folder https://drive.google.com/drive/folders/15d1uva70kgNMi5yClP6tsP6DLv6uzr5_?usp=drive_link```

## CT Experiments
To reproduce the BW-ReLU experiment for the CT reconstruction task (Figure 4) run the following:

``python exp_cts.py -af='bspline-w' --c=3 --lr=2e-3 --layers=3 --width=300 --epochs=10000
--path-norm``

## Signal Representation
To reproduce the BW-ReLU experiment for the signal representation task (Figure 5) run the following:

``python exp_signal_rep.py -af='bspline-w' --c=9 --lr=3e-3 --layers=3 --width=300``
## Superresolution Experiments
To reproduce the BW-ReLU experiment for the superresolution task (Figure 6) run the following:

``python exp_superres.py -af='bspline-w' --c=3 --lr=3e-3 --layers=3 --width=256 --path-norm``

