import torch
import torch.nn.functional as F
import random
from models.MotionLenEstimator import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
import codecs as cs
from sklearn.metrics import mean_absolute_error, r2_score
import torch.nn as nn
from datetime import datetime
import gc
import clip
import spacy  # assuming spacy is required for NLP processing
from ..MotionLenEstimator.LengthEst import *
import torch.distributed as dist


class Logger(object):
    def __init__(self, log_dir):
        pass

    def scalar_summary(self, tag, value, step):
        pass

def reduce_loss(loss, world_size):
    """Reduce loss across all processes and calculate average."""
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= world_size
    return loss

class LengthEstTrainer_V2(object):

    def __init__(self, args, estimator):
        self.opt = args
        self.estimator = estimator
        self.device = args.device
        self.first_save = True
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = pjoin(self.opt.model_dir, self.start_time)
        model_path = "/path/to/spacy/en_core_web_sm"
        self.nlp = spacy.load(model_path)
        self.clip, _ = clip.load('ViT-B/32', self.device)
        self.clip.to(self.device)
        set_requires_grad(self.clip, False)
        os.makedirs(self.save_dir, exist_ok=True)

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()
            self.MSE_loss = torch.nn.MSELoss()
            self.L1_loss = torch.nn.L1Loss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, epoch, niter, val_loss=None, l1_loss=None):
        state = {
            'estimator': self.estimator.state_dict(),
            'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        model_path = pjoin(self.save_dir, 'finest_1.tar')
        torch.save(state, model_path)

        txt_file = pjoin(self.save_dir, 'training_info.txt')
        if self.first_save:
            with open(txt_file, 'w') as f:
                for key, value in vars(self.opt).items():
                    f.write(f'{key}: {value}\n')
                f.write('-' * 80 + '\n')
                f.write('\n')
            self.first_save = False

        if val_loss is not None and l1_loss is not None:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            with open(txt_file, 'w') as f:
                f.writelines(lines[:-1])
                f.write(f'Epoch: {epoch}, Iter: {niter}, Val Loss: {val_loss:.5f}, L1 Loss: {l1_loss:.5f}\n')

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            nn.utils.clip_grad_norm_(network.parameters(), 0.1)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def encode_text(self, raw_text):
        text = clip.tokenize(raw_text, truncate=True).to(self.device)
        self.clip.token_embedding = self.clip.token_embedding.to(self.device)
        feat_clip_text = self.clip.token_embedding(text).type(torch.float32)
        return feat_clip_text

    def get_dependency_edges(self, sentences):
        dependency_edges = []
        token_counts = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            edges = [(token.i, token.head.i) for token in doc if token.head.i != token.i]
            dependency_edges.append(edges)
            token_counts.append(len(doc))
        return dependency_edges, token_counts

    def build_semantic_graph(self, sentences, device, max_len=77):
        dependency_edges, token_counts = self.get_dependency_edges(sentences)
        feat_clip_text = self.encode_text(sentences).to(self.device)
        batch_size = len(sentences)

        x_padded = torch.zeros((batch_size, max_len, 512), device=device)
        edge_indices = []

        for i, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            n_tokens = token_counts[i]
            x = feat_clip_text[i, :n_tokens, :]

            if n_tokens < max_len:
                pad = torch.zeros((max_len - n_tokens, 512), device=device)
                x = torch.cat([x, pad], dim=0)

            x_padded[i] = x
            edge_index = torch.tensor(dependency_edges[i], dtype=torch.long).t().contiguous().to(device)
            edge_indices.append(edge_index)
        return x_padded, edge_indices

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.opt_estimator = optim.AdamW(self.estimator.parameters(), lr=self.opt.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_estimator, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        val_loss = 0
        min_l1_loss = float('inf')
        logs = OrderedDict({'loss': 0})

        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                print(f"Processing training batch {i}")
                self.estimator.train()
                split_len_, word_emb, pos_ohot, cap_lens, total_caption, total_step = batch_data
                split_len_ = split_len_.to(self.device)
                word_emb = word_emb.to(self.device)
                pos_ohot = pos_ohot.to(self.device)
                total_step = total_step.to(self.device)
                word_emb = word_emb.detach().to(self.device).float()
                pos_ohot = pos_ohot.detach().to(self.device).float()

                loss = torch.tensor(0., device=self.device)
                l1_loss = torch.tensor(0., device=self.device)
                self.zero_grad([self.opt_estimator])

                # 计算当前批次大小
                batch_size = split_len_.size(0)

                for j in range(self.opt.mul_data_size):
                    print(total_caption[j])
                    pred_dis = self.estimator(total_caption[j], cap_lens[j], total_step).to(self.device) * self.opt.max_motion_length
                    gt_labels = split_len_[:, j].float().to(self.device)
                    l1_loss += self.L1_loss(pred_dis, gt_labels) / batch_size  # 平均 L1 损失
                    loss += self.MSE_loss(pred_dis, gt_labels) / batch_size  # 平均 MSE 损失

                l1_loss.backward()

                # 同步损失并取平均
                l1_loss = reduce_loss(l1_loss, self.opt.world_size)

                # 仅在rank=0上记录日志
                if self.opt.rank == 0:
                    logs['loss'] += l1_loss.item()
                    print(f"Rank {self.opt.rank} - Iter {i}, Loss: {loss.item()}")

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])
                it += 1

                # torch.cuda.empty_cache()
                # gc.collect()

                if it % self.opt.log_every == 0 and self.opt.rank == 0:
                    self.logger.scalar_summary('loss', logs['loss'] / self.opt.log_every, it)
                    logs = OrderedDict({'loss': 0})

            epoch += 1
            # torch.cuda.empty_cache()
            # gc.collect()

            if epoch % self.opt.save_every_e == 0 and self.opt.rank == 0:
                self.save(epoch, it)

            print('Validation time:')
            val_loss = torch.tensor(0., device=self.device)
            l1_loss = torch.tensor(0., device=self.device)

            with torch.no_grad():
                self.estimator.eval()
                for i, batch_data in enumerate(val_dataloader):
                    split_len_, word_emb, pos_ohot, cap_lens, total_caption, total_step = batch_data
                    split_len_ = split_len_.to(self.device)
                    word_emb = word_emb.to(self.device)
                    pos_ohot = pos_ohot.to(self.device)
                    total_step = total_step.to(self.device)

                    for j in range(self.opt.mul_data_size):
                        pred_dis = self.estimator(total_caption[j], cap_lens[j], total_step).to(self.device) * self.opt.max_motion_length
                        gt_labels = split_len_[:, j].float().to(self.device)
                        val_loss += self.MSE_loss(pred_dis, gt_labels)
                        l1_loss += self.L1_loss(pred_dis, gt_labels)

                # 同步验证损失
                val_loss = reduce_loss(val_loss, self.opt.world_size)
                l1_loss = reduce_loss(l1_loss, self.opt.world_size)
            print('len(val_dataloader):',len(val_dataloader))
            # 除以验证集的批次数
            val_loss /= len(val_dataloader)
            l1_loss /= len(val_dataloader)

            if self.opt.rank == 0:
                print(f"Validation Loss: {val_loss.item()}, L1 Loss: {l1_loss.item()}")
                if l1_loss < min_l1_loss:
                    self.save(epoch, it, val_loss.item(), l1_loss.item())
                    min_l1_loss = l1_loss.item()
