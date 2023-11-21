import enum


class ConfigEnum(enum.Enum):
    PREDICT='getPredict'
    UPLOAD='sendFile'
    ORDER='excuteOrder'
    BASEPATH='http://5301.ETVP.TECH:7788/'
    DATA='sendData'

print(ConfigEnum.UPLOAD.value)