import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric

class EnhancemetPipline(pl.LightningModule):
    def __init__(self, enhacement, asr_encoder, stft_layer, learnig_rate):
        super().__init__()
        
        self.enhacement = enhacement
        self.asr_encoder = asr_encoder
        self.stft_layer = stft_layer
        self.learning_rate = learnig_rate
        self.loss_fn = self._loss_fn()
    
    def forward(self, noisy_input, clean_target, length_ratio):

        noisy_stft = self.stft_layer(noisy_input).permute([0,3,1,2])
        clean_stft = self.stft_layer(clean_target)

        esti_list = self.enhacement(noisy_stft)
        enhancement_output = esti_list[-1].permute([0,3,2,1])

        noisy_embed = self.asr_encoder(enhancement_output, length_ratio)
        target_embed = self.asr_encoder(clean_stft, length_ratio)

        return noisy_embed, target_embed, clean_stft.permute(0,3,1,2), esti_list
    
    def training_step(self, batch, batch_idx):
        
        loss_asr, loss_enh = self._common_step(batch, batch_idx)
        loss = loss_asr + loss_enh

        self.log_dict(
            {
                "train_total_loss": loss,
                "train_ASR_loss": loss_asr,
                "train_Enh_loss": loss_enh,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {'loss':loss}



    def validation_step(self, batch, batch_idx):
        
        loss_asr, loss_enh = self._common_step(batch, batch_idx)
        loss = loss_asr + loss_enh

        self.log("val_total_loss", loss)
        self.log("val_ASR_loss", loss_asr)
        self.log("val_Enh_loss", loss_enh)

        return loss

    
    def test_step(self, batch, batch_idx):

        loss_asr, loss_enh = self._common_step(batch, batch_idx)
        loss = loss_asr + loss_enh

        self.log("test_total_loss", loss)
        self.log("test_ASR_loss", loss_asr)
        self.log("test_Enh_loss", loss_enh)

        return loss

    def _common_step(self, batch, batch_idx):
        
        noisy_input, clean_target, length_ratio = batch
        noisy_embeds, target_embeds, target_stft, gag_list = self.forward(noisy_input, clean_target, length_ratio)
        
        loss_enh = self.loss_fn['enh'](gag_list, target_stft, target_stft.shape[-1])
        loss_asr = self.loss_fn['asr'](noisy_embeds, target_embeds)

        return loss_asr, loss_enh

    def predict_step(self, batch, batch_idx):
        pass

    def _loss_fn(self):

        def asr_loss_fn(noisy_embeds, clean_embeds):
            loss = torch.abs(1.0 - F.cosine_similarity(noisy_embeds, clean_embeds, dim=2))
            return torch.mean(loss)

        def enh_loss_fn(esti_list, label, seq_len):
            frame_list = []
            for i in range(BATCH_SIZE):
                    frame_list.append(seq_len)
            alpha_list = [0.1 for _ in range(len(esti_list))]
            alpha_list[-1] = 1
            mask_for_loss = []
            utt_num = label.size()[0]
            with torch.no_grad():
                for i in range(utt_num):
                    tmp_mask = torch.ones((frame_list[i], label.size()[-2]), dtype=label.dtype)
                    mask_for_loss.append(tmp_mask)
                mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
                mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
                com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
            loss1, loss2 = 0., 0.
            mag_label = torch.norm(label, dim=1)
            for i in range(len(esti_list)):
                mag_esti = torch.norm(esti_list[i], dim=1)
                loss1 = loss1 + alpha_list[i] * (((esti_list[i] - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
                loss2 = loss2 + alpha_list[i] * (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
            return 0.5 * (loss1 + loss2)
        
        return {'asr': asr_loss_fn, 'enh': enh_loss_fn} 
    
    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.learning_rate, rho=0.95, eps=1e-08)
