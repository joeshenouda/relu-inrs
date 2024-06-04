# relu-inrs

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

