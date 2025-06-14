import torch.nn as nn
import torch.nn.functional as F
from segment_anything import SamPredictor
class Model(nn.Module):

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model

    def setup(self):
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder: # unfreeze only mask_decoder
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images) # get image_embeddings by loading image to image_encoder
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder( # get sparse_embeddings, dense_embeddings by loading bboxes to prompt_encoder
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0), 
                image_pe=self.model.prompt_encoder.get_dense_pe(), # Returns the positional encoding used to encode point prompts, applied to a dense set of points the shape of the image encoding.
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, # ouput only one mask per objects in an image
            )

            masks = F.interpolate( # low_res_masks upsampling
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)