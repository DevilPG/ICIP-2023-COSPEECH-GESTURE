from core.networks.keypoints_generation.generator import SequenceGeneratorCNN, HierarchicalPoseGenerator
from core.networks.keypoints_generation.discriminator import PoseSequenceDiscriminator, LipSequenceDiscriminator, HandDiscriminator
from core.networks.poses_reconstruction.autoencoder import Autoencoder, PoseSeqEncoder
from core.networks.joint2encoder import Jointencoder


module_dict = {
    'SequenceGeneratorCNN': SequenceGeneratorCNN,
    'PoseSequenceDiscriminator': PoseSequenceDiscriminator,
    'LipSequenceDiscriminator': LipSequenceDiscriminator,
    'HandDiscriminator': HandDiscriminator,
    'Autoencoder': Autoencoder,
    'PoseSeqEncoder': PoseSeqEncoder,
    'Jointencoder':Jointencoder,
    'HierarchicalPoseGenerator': HierarchicalPoseGenerator
}


def get_model(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown model: %s' % name)
    else:
        return obj
