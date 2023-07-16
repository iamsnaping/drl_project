import os
import sys

print(sys.path)
os.chdir(sys.path[0])
sys.path.append('/home/wu_tian_ci/drl_project/mymodel/envdataset')

sys.path.append('/home/wu_tian_ci/drl_project/mymodel/pretrain_model')