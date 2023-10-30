from typing import Any
import torch.nn as nn

class MLMLoss():
  def __init__(self, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
    self.loss_weights = loss_weights
    self.gen_loss_fc = nn.CrossEntropyLoss(label_smoothing=gen_label_smooth) if gen_label_smooth else nn.CrossEntropyLoss()
    self.disc_loss_fc = nn.BCEWithLogitsLoss()
    self.disc_label_smooth = disc_label_smooth
    
  def __call__(self, pred, targets):
    mlm_gen_logits, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
    gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targets[is_mlm_applied])
    disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
    is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
    if self.disc_label_smooth:
      is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)
    disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
    return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]

class NSPLoss():
    def __init__(self) -> None:
        self.fn = nn.BCEWithLogitsLoss()
    def __call__(self, pred, target) -> Any:
        return self.fn(pred, target)