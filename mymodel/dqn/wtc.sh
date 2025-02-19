python dqntrainallscene.py -cuda cuda:0 -lr 0.001 -restrict True -sup fullscene_30epochs -batchsize 256 -epochs 30 -padding True -appF True -appP pre
python mutidqniteration.py -cuda cuda:0 -lr 0.00005 -epochs 30 -restrict True -preload True -sup restrict_iteration_fulscene -preload True -padding True

