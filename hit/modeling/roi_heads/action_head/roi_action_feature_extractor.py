import torch
from torch import nn
from torch.nn import functional as F

from hit.modeling import registry
from hit.modeling.poolers import make_3d_pooler
from hit.modeling.roi_heads.action_head.hit_structure import make_hit_structure
from hit.modeling.utils import cat, pad_sequence, prepare_pooled_feature
from hit.utils.IA_helper import has_object, has_hand
from hit.structures.bounding_box import BoxList

from hit.modeling.roi_heads.action_head.pose_transformer import PoseTransformer
from hit.modeling.poolers import Pooler3d

def create_square_from_center(x_center, y_center, side_length, img_size, device):
    half_side = torch.tensor(side_length / 2).to(device)
    w, h = img_size
    w = torch.tensor(w).to(device)
    h = torch.tensor(h).to(device)
    zero_tensor = torch.tensor(0).to(device)
    x0 = torch.max(zero_tensor, x_center - half_side)
    y0 = torch.max(zero_tensor, y_center - half_side)
    x1 = torch.min(w, x_center + half_side)
    y1 = torch.min(h, y_center + half_side)
    return x0, y0, x1, y1

@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD
        resolution = head_cfg.POOLER_RESOLUTION
        self.pooler = make_3d_pooler(head_cfg)
        slow_fast_pooler = [0.1, 0.0625, 0.04]
        x3d_pooler = [0.05, 0.03125, 0.02]
        choosen_pooler = slow_fast_pooler
        self.pooler2 = Pooler3d(
                        output_size=(resolution, resolution),
                        scale=choosen_pooler[0],
                        sampling_ratio=0,
                        pooler_type="align3d",
                    )
        self.pooler3 = Pooler3d(
                output_size=(resolution, resolution),
                scale=choosen_pooler[-1],
                sampling_ratio=0,
                pooler_type="align3d",
            )

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))
        in_channels =out_channel = representation_size = 256
        self.convT = nn.Sequential(nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=(4, 1, 1),  # Kernel size for time (4), spatial (1x1)
            stride=(2, 1, 1),       # Stride for time (2), spatial (1x1)
            padding=(1, 0, 0),      # Padding for time (1), spatial (0x0)
            bias=True),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fcb1 = nn.Sequential(
            nn.Linear(out_channel * 3 * 16 * 7 * 7, representation_size),
            nn.BatchNorm1d(representation_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fcbhands = nn.Sequential(
            nn.Linear(out_channel * 3 * 8 * 7 * 7, representation_size),
            nn.BatchNorm1d(representation_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.proj_hands = nn.Linear(config.MODEL.HIT_STRUCTURE.DIM_OUT * 2, config.MODEL.HIT_STRUCTURE.DIM_OUT)
        self.pose_out = config.MODEL.HIT_STRUCTURE.DIM_INNER
        self.hit_structure_of = None
        if config.MODEL.HIT_STRUCTURE.ACTIVE:
            self.max_feature_len_per_sec = config.MODEL.HIT_STRUCTURE.MAX_PER_SEC
            self.hit_structure = make_hit_structure(config, dim_in)

        fc1_dim_in = dim_in
        if config.MODEL.HIT_STRUCTURE.ACTIVE and (config.MODEL.HIT_STRUCTURE.FUSION == "concat"):
            fc1_dim_in += config.MODEL.HIT_STRUCTURE.DIM_OUT

        self.fc2 = nn.Linear(representation_size, representation_size)

        for l in [self.proj_hands, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        self.dim_out = representation_size

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def roi_3pooling(self, fast_features, proposals):
        x1 = self.pooler(fast_features, proposals)
        x2 = self.pooler2(fast_features, proposals)
        x3 = self.pooler3(fast_features, proposals)
        person_pooled = torch.cat([x1, x2, x3], dim=1)
        return person_pooled

    def max_pooling_zero_safe(self, x):
        if x.size(0) == 0:
            _, c, t, h, w = x.size()
            res = self.config.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION
            x = torch.zeros((0, c, 1, h - res + 1, w - res + 1), device=x.device)
        else:
            x = self.max_pooler(x)
        return x

    def forward(self, slow_features, fast_features, proposals, objects=None, keypoints=None, extras={}, part_forward=-1, of_features=None):
        ia_active = hasattr(self, "hit_structure")
        if part_forward == 1:
            person_pooled = cat([box.get_field("pooled_feature") for box in proposals])
            if objects is None or any([x is None for x in objects]):
                object_pooled = None
            else:
                object_pooled = cat([box.get_field("pooled_feature") for box in objects])
                
            if keypoints is None:
                hands_pooled = None
            else:
                hands_pooled = cat([box.get_field("pooled_feature") for box in keypoints[0]])
                # pose_out = cat([box.get_field("pooled_feature") for box in keypoints[1]])
                pose_out = None
                keypoints = keypoints[0]
        else:
            fast_features_ = self.convT(fast_features)
            person_pooled = self.roi_3pooling(fast_features_, proposals)
            person_pooled = person_pooled.view(person_pooled.size(0), -1) #bs * 256 x 8  
            person_pooled = self.fcb1(person_pooled)     # Couche linéaire
            person_pooled = person_pooled[..., None, None, None]

            if has_object(self.config.MODEL.HIT_STRUCTURE):
                object_pooled = self.roi_pooling(slow_features, fast_features, objects)
                object_pooled = self.max_pooling_zero_safe(object_pooled)
                # TODO : same size object_pooled and person_pooled bs x 2304 
                # si que 8 / 32 alors comment fait il le mapping? a qui appartient les 8?
            else:
                object_pooled = None
            hand_boxlists = []

            if has_hand(self.config.MODEL.HIT_STRUCTURE):    
                for k in keypoints:
                    if 'keypoints' in k.extra_fields:
                        kk = torch.flatten(k.extra_fields['keypoints'], start_dim=1)[:, 18:22]
                        #TODO : test this.
                        x0 = kk[:,[0,2]].min(dim=1).values
                        y0 = kk[:,[1,3]].min(dim=1).values
                        x1 = kk[:,[0,2]].max(dim=1).values
                        y1 = kk[:,[1,3]].max(dim=1).values
                        kk = torch.stack((x0, y0, x1, y1), dim=1)
                        hand_boxlists.append(BoxList(kk, k.size, mode="xyxy", dtype=k.bbox.dtype))
                    else:
                        hand_boxlists.append(BoxList(torch.zeros((0, 4), dtype=k.bbox.dtype, device=k.bbox.device), k.size, mode="xyxy"))
                proposals_hand = [box.extend((0.2, 0.8)) for box in hand_boxlists]

                hands_pooled = self.roi_3pooling(fast_features, proposals_hand)
                hands_pooled = hands_pooled.view(hands_pooled.size(0), -1) #bs * 256 x 8  
                hands_pooled = self.fcbhands(hands_pooled)
                hands_pooled = hands_pooled[..., None, None, None]
            
                
            else:
                hands_pooled = None
            
            # else:
            pose_out = None
            # Pose end
                
        if part_forward == 0:
            return None, person_pooled, object_pooled, hands_pooled, pose_out

        x_after = person_pooled
        if ia_active:
            tsfmr = self.hit_structure
            mem_len = self.config.MODEL.HIT_STRUCTURE.LENGTH
            mem_rate = self.config.MODEL.HIT_STRUCTURE.MEMORY_RATE
            use_penalty = self.config.MODEL.HIT_STRUCTURE.PENALTY
            memory_person = None
            if "M" in self.config.MODEL.HIT_STRUCTURE.I_BLOCK_LIST:
                memory_person, memory_person_boxes = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                                                        self.max_feature_len_per_sec, tsfmr.dim_others,
                                                                        person_pooled, proposals, use_penalty)
                # RGB stream
            ia_feature, res_person, res_object, res_keypoint = self.hit_structure(person_pooled, proposals, object_pooled, objects, hands_pooled, keypoints, memory_person, None, None, phase="rgb")
            x_after = self.fusion(x_after, ia_feature, self.config.MODEL.HIT_STRUCTURE.FUSION)
        x_after = x_after.view(x_after.size(0), -1)
        
        x_after = F.relu(self.fc2(x_after))

        return x_after, person_pooled, object_pooled, hands_pooled, pose_out

    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, fixed_dim, current_x, current_box, use_penalty):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        if use_penalty and self.training:
            cur_loss = extras["cur_loss"]
        else:
            cur_loss = 0.0
        current_feat = prepare_pooled_feature(current_x, current_box, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            # mem rate is a parameter in config file : it indicates the amount of timestamp you can go look back or after to help predict this one.
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = feature_pool[movie_id]
            
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                   for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                  for mem_ind in after_inds]
            mem_box_current = [self.sample_mem_feature(new_feat, max_boxes), ]
            # mem_box_list_before = mem_box_list_after = [None for  mem_ind in after_inds] # HACK TO TEST WITHOUT MEMORY
            mem_box_list = mem_box_list_before + mem_box_current + mem_box_list_after
            mem_feature_list += [box_list.get_field("pooled_feature")
                                 if box_list is not None
                                 else torch.zeros(0, fixed_dim, 1, 1, 1, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]

        seq_length = sum(mem_len) + 1
        person_per_seq = seq_length * max_boxes
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, fixed_dim, 1, 1, 1)
        mem_feature = mem_feature.to(device)
        mem_pos = pad_sequence(mem_pos_list, max_boxes)
        mem_pos = mem_pos.view(-1, person_per_seq, 4)
        mem_pos = mem_pos.to(device)

        return mem_feature, mem_pos

    def check_fetch_mem_feature(self, movie_cache, mem_ind, max_num, cur_loss, use_penalty):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        box_list = self.sample_mem_feature(box_list, max_num)
        if use_penalty and self.training:
            loss_tag = box_list.delete_field("loss_tag")
            penalty = loss_tag / cur_loss if loss_tag < cur_loss else cur_loss / loss_tag
            features = box_list.get_field("pooled_feature") * penalty
            box_list.add_field("pooled_feature", features)
        return box_list

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError


def make_roi_action_feature_extractor(cfg, dim_in):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, dim_in)
