from core.pipelines.voice2pose import Voice2Pose
from core.pipelines.pose2pose import Pose2Pose
from core.pipelines.joint2pose import Joint2Pose


module_dict = {
    'Voice2Pose': Voice2Pose,
    'Pose2Pose': Pose2Pose,
    'Joint2Pose': Joint2Pose,
}


def get_pipeline(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown pipeline: %s' % name)
    else:
        return obj
