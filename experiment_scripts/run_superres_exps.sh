python exp_superres.py -af='bspline-w' --width=256 --device=0 --lr=3e-3 --c=3 --layers=3 --lr-decay=0.8 --epochs=2000 --lam=0 --path-norm --rand-seed=40
python exp_superres.py -af='wire' --width=256 --device=0 --lr=1e-2 --omega0=8 --sigma0=6 --lr-decay=0.2 --epochs=2000 --lam=0 --rand-seed=40
