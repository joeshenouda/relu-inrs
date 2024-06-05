## c=3 Experiment
python exp_ct.py -af='bspline-w' --c=3 --lr=3e-3 --epochs=10000 --layers=3 --width=300  --path-norm --rand-seed=35

## c=5 Experiment
python exp_ct.py -af='bspline-w' --c=5 --lr=3e-3 --epochs=10000 --layers=3 --width=300 --path-norm
--rand-seed=35 --stop-loss=4.73e-2

## c=2 Experiment
python exp_ct.py -af='bspline-w' --c=2 --lr=3e-3 --epochs=100000 --layers=3 --width=300 --path-norm
--rand-seed=35 --stop-loss=4.73e-2

## c=1 Experiment
python exp_ct.py -af='bspline-w' --c=1 --lr=3e-3 --epochs=100000 --layers=3 --width=300 --path-norm
--rand-seed=35 --stop-loss=4.73e-2
