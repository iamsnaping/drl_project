import enum
import os
class OnlineConfig(enum.Enum):
    #experiment1 afternoon
    LOAD_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20240104/model/090704/onlineModel.pt'
    ITERATION_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20240104/model/090704/onlineModel.pt'
    NO_ITERATION_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20240104/model/090704/onlineModel.pt'
    
    # # /home/wu_tian_ci/single
    # # experiment
    EYE_DATA_PATH='/home/wu_tian_ci/dataset/experiment'
    ONLIEN_DATA_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata'
    INDIVIDUAL='/home/wu_tian_ci/dataset/experiment/23/1'
    INDIVIDUAL_NO_SCENE='/home/wu_tian_ci/dataset/experiment/23'
    SCENE='1'
    DATASET_PATH='/home/wu_tian_ci/dataset/experiment'

    # experiment2 morning
    # LOAD_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20240105/model/163017/onlineModel.pt'
    # ITERATION_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/202401;05/model/163017/onlineModel.pt'
    # NO_ITERATION_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20240105/model/163017/onlineModel.pt'
    
    # /home/wu_tian_ci/single
    # experiment
    # EYE_DATA_PATH='/home/wu_tian_ci/dataset/experiment2'
    # ONLIEN_DATA_PATH='/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata'
    # INDIVIDUAL='/home/wu_tian_ci/dataset/experiment2/23/1'
    # INDIVIDUAL_NO_SCENE='/home/wu_tian_ci/dataset/experiment2/23'
    # SCENE='1'
    # DATASET_PATH='/home/wu_tian_ci/dataset/experiment2'




    SKIP_LIST=[7,12,13]
    SKIP=False
    PERSON_ENVS=23
    LR=0.00005
    PERSON=23
    RESTRICT=True
    RESTRICT_AREA=[0,1,2,3,6,9]


# print(len(os.listdir(OnlineConfig.EYE_DATA_PATH.value)))