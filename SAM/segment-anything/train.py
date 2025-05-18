import torch
import os
from utils import *
from segment_anything import sam_model_registry
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# load SAM checkpoint 
sam = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint)

model = Model(cfg, sam).to(device)
model.setup()

train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)

# optimizer = torch.optim.Adam(model.model.parameters())
# In the paper, the authors used AdamW for training.
optimizer = torch.optim.AdamW(model.model.parameters(), lr=cfg.opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.opt.weight_decay, amsgrad=False)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_sam(cfg, model, optimizer, scheduler, train_data, val_data, device)\

state_dict = model.model.state_dict()
torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{1}-loss-{81.41}-ckpt.pth"))