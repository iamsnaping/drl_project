import enum
import os
class OnlineConfig(enum.Enum):
    LOAD_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231225/trainallscene/1/dqnnetoffline.pt'
    ITERATION_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231225/trainallscene/1/dqnnetoffline.pt'
    NO_ITERATION_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231225/trainallscene/1/dqnnetoffline.pt'
    EYE_DATA_PATH='/home/wu_tian_ci/smallbatch'
    ONLIEN_DATA_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata'
    # INDIVIDUAL='/home/wu_tian_ci/eyedatanew/23/1'
    # INDIVIDUAL_NO_SCENE='/home/wu_tian_ci/eyedatanew/23'
    INDIVIDUAL='/home/wu_tian_ci/smallbatch/23/1'
    INDIVIDUAL_NO_SCENE='/home/wu_tian_ci/smallbatch/23'
    SCENE='1'
    DATASET_PATH='/home/wu_tian_ci/smallbatch/'
    SKIP_LIST=[7,12,13]
    SKIP=False
    PERSON_ENVS=23
    LR=0.00005
    PERSON=23


# print(len(os.listdir(OnlineConfig.EYE_DATA_PATH.value)))