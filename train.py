import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd
import numpy as np

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

import random
import traceback
import sys

from utils import Train_Report
from model.dit import DiT


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_data_loader = data_loader['test']

        # Accelerator
        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process:
            if args.work_dir is not None:
                os.makedirs(args.work_dir, exist_ok=True)

        # Weight data type
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Model
        self.load_checkpoint()
        self.text_prompt()
        self.noise = torch.randn(self.args.num_noise, 1, 1, self.args.in_channels).to(self.accelerator.device)
        self.dit = DiT(in_channels=self.args.in_channels, hidden_size=self.args.hidden_size, depth=self.args.depth, num_heads=self.args.num_heads)

        self.unseen_num = self.args.unseen_label
        self.unseen_labels = np.load(self.args.unseen_label_path)

        if 'ntu60' in self.args.unseen_label_path:
            self.seen_labels = np.where(~np.isin(np.arange(60), self.unseen_labels))[0]
        else:  # ntu120
            self.seen_labels = np.where(~np.isin(np.arange(120), self.unseen_labels))[0]

        self.seen_num = len(self.seen_labels)

        self.seen_label_mapping = {label: idx for idx, label in enumerate(self.seen_labels)}
        self.unseen_label_mapping = {label: idx for idx, label in enumerate(self.unseen_labels)}

        # Optimizer
        params_to_opt = self.dit.parameters()
        self.optimizer = torch.optim.AdamW(params_to_opt, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        # Scheduler
        self.lr_scheduler = get_scheduler(self.args.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=self.args.num_warmup, num_training_steps=self.args.num_iter)

        self.dit, self.optimizer, self.lr_scheduler, self.train_data_loader, self.val_data_loader, self.test_data_loader = self.accelerator.prepare(
            self.dit, self.optimizer, self.lr_scheduler, self.train_data_loader, self.val_data_loader, self.test_data_loader)

    def load_checkpoint(self):
        noise_scheduler_config = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler").config
        noise_scheduler_config['num_train_timesteps'] = self.args.num_steps
        noise_scheduler_config['prediction_type'] = self.args.prediction_type
        self.noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.requires_grad_(False)

    def text_prompt(self):
        text_embed_csv = []
        df = pd.read_csv('./data/class_lists/ntu120.csv')  # 'ntu120.csv' includes 'ntu60.csv'
        prompts = df['label'].values.tolist()
        for prompt in prompts:
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=35, truncation=True, return_tensors="pt").to(self.accelerator.device)
            embeds = self.text_encoder(text_inputs.input_ids)
            prompt_embeds = embeds['last_hidden_state']
            pooled_prompt_embeds = embeds['pooler_output']
            text_embed_csv.append(torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone().to(self.accelerator.device))

        text_embed_txt = []
        with open('./data/class_lists/ntu120_llm.txt', 'r', encoding='utf-8') as file:  # 'ntu120_llm.txt' includes 'ntu60_llm.txt'
            content_list = file.readlines()
        prompts = [line.strip() for line in content_list]
        for prompt in prompts:
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=35, truncation=True, return_tensors="pt").to(self.accelerator.device)
            embeds = self.text_encoder(text_inputs.input_ids)
            prompt_embeds = embeds['last_hidden_state']
            pooled_prompt_embeds = embeds['pooler_output']
            text_embed_txt.append(torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone().to(self.accelerator.device))

        self.text_embed = torch.cat((torch.cat(text_embed_csv, dim=0), torch.cat(text_embed_txt, dim=0)), dim=-1)

    def save_best_model(self):
        save_path = os.path.join(self.args.work_dir, 'best')
        self.accelerator.save_state(save_path)

    def train(self, train_log, global_step):
        self.dit.train()
        self.dit.requires_grad_(True)
        report = Train_Report()
        start = time.time()

        for idx, (features, labels) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.dit):
                with torch.no_grad():
                    features = features.to(self.accelerator.device, dtype=self.weight_dtype).unsqueeze(dim=1)
                    text_embed = self.text_embed[labels.long()].to(self.accelerator.device, dtype=self.weight_dtype)
                    noise = torch.randn(features.shape, device=features.device).repeat(2, 1, 1)
                    timesteps = torch.randint(0, self.args.num_steps, (self.args.batch_size,), device=features.device).long().repeat(2)

                    labels_r = torch.Tensor(random.choices(self.seen_labels, k=self.args.batch_size)).to(self.accelerator.device)
                    text_embed_r = self.text_embed[labels_r.long()].to(self.accelerator.device, dtype=self.weight_dtype)

                    mask = (labels != labels_r).to(self.accelerator.device, dtype=self.weight_dtype)

                    features = features.repeat(2, 1, 1)
                    text_embed = torch.cat((text_embed, text_embed_r), dim=0)
                    noisy_latents = self.noise_scheduler.add_noise(features, noise, timesteps)
                
                model_pred = self.dit(noisy_latents, timesteps, text_embed[:, -1, :], text_embed[:, :-1, :])

                if "sample" == self.args.prediction_type:
                    target = features
                elif "epsilon" == self.args.prediction_type:
                    target = noise
                elif "v_prediction" == self.args.prediction_type:
                    target = self.noise_scheduler.get_velocity(features, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.args.prediction_type}")

                loss_diff = F.mse_loss(model_pred[:self.args.batch_size], target[:self.args.batch_size]) * self.args.d_weight

                pos = F.mse_loss(model_pred[:self.args.batch_size], target[:self.args.batch_size], reduce=False).mean(dim=(1, 2))
                neg = F.mse_loss(model_pred[self.args.batch_size:], target[self.args.batch_size:], reduce=False).mean(dim=(1, 2))
                triplet = torch.clamp(pos - neg + self.args.margin, min=0.0) * mask
                loss_triplet = triplet.mean() * self.args.t_weight

                loss = loss_diff + loss_triplet

                reduced_loss = self.accelerator.gather(loss).mean()
                reduced_loss_diff = self.accelerator.gather(loss_diff).mean()
                reduced_loss_triplet = self.accelerator.gather(loss_triplet).mean()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size, reduced_loss.item(), reduced_loss_diff.item(), reduced_loss_triplet.item())

            global_step += 1

            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                result_str = report.result_str(lr, period_time)
                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

            if global_step % self.args.save_iter == 0:
                save_path = os.path.join(self.args.work_dir, f'checkpoint-{global_step}')
                self.accelerator.save_state(save_path)

        return global_step

    def test(self):
        self.dit.eval()
        self.dit.requires_grad_(False)
        cnt = 0
        num = 0

        with torch.no_grad():
            for idx, (features, labels) in tqdm(enumerate(self.test_data_loader)):
                label_pred = torch.zeros((features.shape[0], self.unseen_num)).to(self.accelerator.device)
                mapped_labels = [self.unseen_label_mapping[label.item()] for label in labels]
                mapped_labels = torch.Tensor(mapped_labels).to(self.accelerator.device).long()

                t = torch.ones((features.shape[0])) * self.args.idx_inference_step
                t = t.to(self.accelerator.device)

                features = features.to(self.accelerator.device, dtype=self.weight_dtype).unsqueeze(dim=1)
                labels = labels.to(self.accelerator.device).long()

                for j in range(self.args.num_noise):
                    noise = self.noise[j].repeat(features.shape[0], 1, 1)
                    noisy_latents = self.noise_scheduler.add_noise(features, noise, t.long())
                    for idx, i in enumerate(self.unseen_labels):
                        labels_ = torch.ones((features.shape[0])).to(self.accelerator.device) * i
                        text_embed = self.text_embed[labels_.long()].to(self.accelerator.device, dtype=self.weight_dtype)
                        noise_pred = self.dit(noisy_latents, t, text_embed[:, -1, :], text_embed[:, :-1, :])
                        if "sample" == self.args.prediction_type:
                            label_pred[:, idx] += F.mse_loss(noise_pred, features, reduce=False).mean(dim=(1, 2))
                        elif "epsilon" == self.args.prediction_type:
                            label_pred[:, idx] += F.mse_loss(noise_pred, noise, reduce=False).mean(dim=(1, 2))
                        elif "v_prediction" == self.args.prediction_type:
                            label_pred[:, idx] += F.mse_loss(noise_pred, self.noise_scheduler.get_velocity(features, noise, t.long()), reduce=False).mean(dim=(1, 2))
                        else:
                            raise ValueError(f"Unknown prediction type {self.args.prediction_type}")

                cnt += torch.sum(torch.argmin(label_pred, -1) == mapped_labels)
                num += len(labels)

        zsl_accuracy = float(cnt) / num
        return zsl_accuracy