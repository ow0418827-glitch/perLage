"""新运行使用精确模型身份；不继承历史实验的身份兼容表。"""


def model_identity_matches(recorded, current):
    return recorded == current
