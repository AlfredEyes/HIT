from torch import nn

from ..backbone import build_backbone
from ..roi_heads.roi_heads_3d import build_3d_roi_heads


class ActionDetector(nn.Module):
    def __init__(self, cfg):
        super(ActionDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_3d_roi_heads(cfg, self.backbone.dim_out)

    def forward(self, slow_video, fast_video, boxes, objects=None, keypoints=None, extras={}, part_forward=-1):

        if part_forward==1:
            slow_features = fast_features = None
        else:
            slow_features, fast_features = self.backbone(slow_video, fast_video)

        slow_features = None
        fast_features = fast_features[-1].permute(0, 4, 1, 2, 3)
        print(fast_features.shape)
        result, detector_losses, loss_weight, detector_metrics = self.roi_heads(slow_features, fast_features, boxes, objects, keypoints, extras, part_forward)

        if self.training:
            return detector_losses, loss_weight, detector_metrics, result

        return result

    def c2_weight_mapping(self):
        if not hasattr(self, "c2_mapping"):
            weight_map = {}
            for name, m_child in self.named_children():
                if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                    child_map = m_child.c2_weight_mapping()
                    for key, val in child_map.items():
                        new_key = name + '.' + key
                        weight_map[new_key] = val
            self.c2_mapping = weight_map
        return self.c2_mapping

def build_detection_model(cfg):
    return ActionDetector(cfg)