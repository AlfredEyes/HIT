import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np #! Ajouté pour les prints, à retirer plus tard
from .hiera123.hiera import hiera_base_16x224, hiera_tiny_16x224, hiera_small_16x224

def resize_positional_embeddings(model, new_input_size, pretrained_state_dict):
    if model.sep_pos_embed:
        print("\nResizing separate positional embeddings:")

        pos_embed_spatial = pretrained_state_dict['pos_embed_spatial']
        print(f"Original spatial embed shape: {pos_embed_spatial.shape}")
        
        # Calcul dimensions
        new_H = new_input_size[1] // model.patch_stride[1]  #? 608/4 = 152
        new_W = new_input_size[2] // model.patch_stride[2]  #? 1088/4 = 272
        orig_size = int(math.sqrt(pos_embed_spatial.shape[1]))  #? sqrt(3136) = 56
        
        print(f"Resizing spatial grid from {orig_size}x{orig_size} to {new_H}x{new_W}")
        
        #& Redimensionnement embeddings 
        pos_embed_spatial = pos_embed_spatial.reshape(1, orig_size, orig_size, -1)
        pos_embed_spatial = pos_embed_spatial.permute(0, 3, 1, 2)  # [1, dim, H, W]
        pos_embed_spatial = F.interpolate(
            pos_embed_spatial,
            size=(new_H, new_W),
            mode='bicubic',
            align_corners=False
        )
        pos_embed_spatial = pos_embed_spatial.permute(0, 2, 3, 1)  # [1, H, W, dim]
        pos_embed_spatial = pos_embed_spatial.reshape(1, new_H * new_W, -1)
        print(f"Resized spatial embed shape: {pos_embed_spatial.shape}")
        
        pretrained_state_dict['pos_embed_spatial'] = pos_embed_spatial

        #& Embeddings temporels
        pos_embed_temporal = pretrained_state_dict['pos_embed_temporal']
        print(f"Original temporal embed shape: {pos_embed_temporal.shape}")
        
        new_T = new_input_size[0] // model.patch_stride[0]  # 8/2 = 4
        
        if new_T != pos_embed_temporal.shape[1]:
            pos_embed_temporal = pos_embed_temporal.permute(0, 2, 1)  # [1, D, T]
            pos_embed_temporal = F.interpolate(
                pos_embed_temporal,
                size=new_T,
                mode='linear', #? 'linear' pour les embeddings temporels, 'bicubic' pour les spatiaux
                align_corners=False
            )
            pos_embed_temporal = pos_embed_temporal.permute(0, 2, 1)  # [1, T, D]
        
        print(f"Final temporal embed shape: {pos_embed_temporal.shape}")
        pretrained_state_dict['pos_embed_temporal'] = pos_embed_temporal

        expected_spatial_tokens = new_H * new_W  # 152 * 272 = 41,344
        print(f"\nValidation:")
        print(f"Expected spatial tokens: {expected_spatial_tokens}")
        print(f"Got spatial tokens: {pos_embed_spatial.shape[1]}")
        print(f"Expected temporal tokens: {new_T}")
        print(f"Got temporal tokens: {pos_embed_temporal.shape[1]}")

        # Vérification des dimensions
        assert pos_embed_spatial.shape[1] == expected_spatial_tokens, "Spatial tokens mismatch"
        assert pos_embed_temporal.shape[1] == new_T, "Temporal tokens mismatch"

    #^ ancien embedding pour 224x224
    else:
        raise NotImplementedError("Only separate positional embeddings are supported")

    return pretrained_state_dict





#? Class utiliser pour inference
class HieraBackbone123(nn.Module):
    def __init__(self, cfg):
        super(HieraBackbone123, self).__init__()

        print("\n=== HieraBackbone123 Initialization ===")
        self.cfg = cfg.clone()
        input_size = (
            8,        # Temporal dimension
            416,      # Height 
            416      # Width 
        )
        print(f"Config input size: {input_size}")
        patch_stride = (2, 4, 4)
        num_patches = (
            input_size[0] // patch_stride[0],
            input_size[1] // patch_stride[1],
            input_size[2] // patch_stride[2]
        )

        print(f"\n=== Patch Configuration ===")
        print(f"Input size: {input_size}")
        print(f"Number of patches: {num_patches}")
        print(f"Total patches: {np.prod(num_patches)}")
        print(f"Patch stride: {patch_stride}")
        print(f"Expected tokens shape: {num_patches}")

        self.model = hiera_small_16x224(
            num_classes=cfg.MODEL.NUM_CLASSES,
            input_size=input_size,
            in_chans=3,
            patch_stride=(2, 4, 4),
            patch_kernel=(3, 7, 7),
            patch_padding=(1, 3, 3),
            sep_pos_embed=True,
            pretrained=False,
            # Configuration spécifique small
            embed_dim=96,
            num_heads=1,
            stages=(1, 2, 11, 2),
            dim_mul=2.0,
            head_mul=2.0
        )

        if cfg.MODEL.WEIGHT:
            print("\n=== Loading Weights ===")
            pretrained_checkpoint = cfg.MODEL.WEIGHT
            print(f"Loading from: {pretrained_checkpoint}")
            state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
            pretrained_state_dict = state_dict.get('model_state', state_dict)

            n_params = len(pretrained_state_dict)
            print(f"Loaded {n_params} parameters")

            # Filtrer les paramètres indésirables
            pretrained_state_dict = {
                k: v for k, v in pretrained_state_dict.items()
                if not (k.startswith('head.') or k.startswith('norm.'))
            }
            print(f"Kept {len(pretrained_state_dict)} parameters after removing head/norm")

            print("\n=== Resizing Position Embeddings ===")
            resize_positional_embeddings(self.model, input_size, pretrained_state_dict)

            # Charger les poids redimensionnés
            self.model.load_state_dict(pretrained_state_dict, strict=False)
            print("Weights loaded successfully")

            # Assigner explicitement les embeddings positionnels redimensionnés au modèle
            if 'pos_embed_spatial' in pretrained_state_dict:
                self.model.pos_embed_spatial = nn.Parameter(pretrained_state_dict['pos_embed_spatial'])
                print(f"Assigned pos_embed_spatial shape: {self.model.pos_embed_spatial.shape}")
            if 'pos_embed_temporal' in pretrained_state_dict:
                self.model.pos_embed_temporal = nn.Parameter(pretrained_state_dict['pos_embed_temporal'])
                print(f"Assigned pos_embed_temporal shape: {self.model.pos_embed_temporal.shape}")

        self.dim_out = 768
        print(f"\n=== Backbone Ready ===")
        print(f"Output dimension: {self.dim_out}")
        print(f"Patch stride: {self.model.patch_stride}")
        print(f"Token spatial shape: {self.model.tokens_spatial_shape}")

    def forward(self, x, lateral_connection=None):
        print("\n=== HieraBackbone123 Forward Pass ===")
        print(f"Input shape: {x.shape}")  # [1, 3, 8, 608, 1088]
    
        try:
            B, C, T_in, H_in, W_in = x.shape
            
            T = T_in // self.model.patch_stride[0]  # 8/2 = 4
            H = H_in // self.model.patch_stride[1]  # 608/4 = 152
            W = W_in // self.model.patch_stride[2]  # 1088/4 = 272
            H_W = H * W
    
            # Patch embedding
            x = self.model.patch_embed(x)
            print(f"After patch embedding: {x.shape}")  # [B, N, D] où N = T*H*W
    
            if self.model.sep_pos_embed:
                # Get embeddings
                pos_embed_spatial = self.model.pos_embed_spatial  # [1, H*W, D]
                pos_embed_temporal = self.model.pos_embed_temporal  # [1, T, D]
    
                # Extend temporal embeddings
                pos_embed_temporal = pos_embed_temporal.unsqueeze(2)  # [1, T, 1, D]
                pos_embed_temporal = pos_embed_temporal.expand(-1, -1, H_W, -1)  # [1, T, H*W, D]
                pos_embed_temporal = pos_embed_temporal.reshape(1, T * H_W, -1)  # [1, T*H*W, D]
    
                # Extend spatial embeddings
                pos_embed_spatial = pos_embed_spatial.unsqueeze(1)  # [1, 1, H*W, D]
                pos_embed_spatial = pos_embed_spatial.expand(-1, T, -1, -1)  # [1, T, H*W, D]
                pos_embed_spatial = pos_embed_spatial.reshape(1, T * H_W, -1)  # [1, T*H*W, D]
    
                # Combine embeddings
                pos_embed = pos_embed_spatial + pos_embed_temporal
                
                if B > 1:
                    pos_embed = pos_embed.expand(B, -1, -1)
                    
                print(f"Combined position embed shape: {pos_embed.shape}")
            else:
                pos_embed = self.model.get_pos_embed()
    
            # Add positional embeddings
            x = x + pos_embed
            print(f"After adding position embedding: {x.shape}")
    
            # Continue forward pass
            x = self.model.unroll(x)
            
            intermediates = []
            for i, block in enumerate(self.model.blocks):
                x = block(x)
                if i in self.model.stage_ends:
                    rerolled = self.model.reroll(x, i)
                    intermediates.append(rerolled)
    
            return x, intermediates
    
        except Exception as e:
            print(f"\nError in HieraBackbone123 forward pass!")
            print(f"Error message: {str(e)}")
            raise

    # @property
    # def module_str(self):
    #     return "Hiera-Base-16x224"