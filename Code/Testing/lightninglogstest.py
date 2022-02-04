import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


# class MyLightningModule(LightningModule):
#     def __init__(self, learning_rate, *args, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()


# all init args were saved to the checkpoint
checkpoint = torch.load(r'C:\Users\soham\Desktop\ML-Stats\lightning_logs\default\version_0\checkpoints\epoch=29-step=899.ckpt')
print(checkpoint["hyper_parameters"])


# model = MyLightingModule.load_from_checkpoint(PATH)
#
# print(model.learning_rate)
# # prints the learning_rate you used in this checkpoint
#
# model.eval()
# y_hat = model(x)

