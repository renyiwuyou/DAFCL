import torch
import numpy as np
import torch.distributions as dist
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        if self.cuda_condition:
            self.model.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.n_whole_level = args.n_whole_level
        self.beta_distribution = dist.Beta(torch.tensor([args.beta]), torch.tensor([args.beta]))

        # Hard Negative Mining
        self.use_hard_negative = getattr(args, 'use_hard_negative',True)
        self.hard_negative_ratio = getattr(args, 'hard_negative_ratio', 0.3)
        self.hn_candidates = getattr(args, 'hn_candidates', 200)

        
        self.use_freq_augment = getattr(args, 'use_freq_augment', True)
        self.freq_mask_ratio = getattr(args, 'freq_mask_ratio', 0.3)
        self.curriculum_epochs = getattr(args, 'curriculum_epochs', 30)
        
        
        self.adaptive_band = getattr(args, 'adaptive_band', True)
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        if self.use_hard_negative:
            print(f"Hard Negative Mining Enabled: ratio={self.hard_negative_ratio}, candidates={self.hn_candidates}")
        if self.use_freq_augment:
            print(f"Frequency Domain Learning Enabled: FreqMix + Adaptive Band Contrast")

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def get_hard_negatives(self, seq_out, pos_ids, n_candidates=200):
        """
        randomly sample one from the top-10 as the negative sample
        """
        batch_size, seq_len, hidden_size = seq_out.shape
        device = seq_out.device

        # Flatten to [batch*seq_len, hidden]
        seq_out_flat = seq_out.view(-1, hidden_size)
        pos_ids_flat = pos_ids.view(-1)

        # Randomly sample candidate negative samples for each position 
        candidates = torch.randint(1, self.args.item_size, (batch_size * seq_len, n_candidates), device=device)

        # Ensure candidates do not contain the positive item itself
        mask = (candidates == pos_ids_flat.unsqueeze(1))
        if mask.any():
            replacement = torch.randint(1, self.args.item_size, (mask.sum(),), device=device)
            candidates[mask] = replacement

        # Get embeddings of candidate negative samples
        cand_emb = self.model.item_embeddings(candidates)

        # Compute similarity
        query = seq_out_flat.unsqueeze(1)
        sim = torch.bmm(query, cand_emb.transpose(1, 2)).squeeze(1)

        _, top_indices = torch.topk(sim, 10, dim=1)  # Take top-10
        rand_idx = torch.randint(0, 10, (top_indices.size(0),), device=device)  # Randomly select
        selected_idx = top_indices.gather(1, rand_idx.unsqueeze(1)).squeeze(1)

        hard_neg_ids = candidates.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
        return hard_neg_ids.view(batch_size, seq_len)

    def frequency_decomposition(self, sequence_output, epoch):
        
        if not self.use_freq_augment:
            return None, None
            
        progress = min(1.0, epoch / self.curriculum_epochs)
        current_mask_ratio = self.freq_mask_ratio * progress
        
        # FFT Transformation
        x_fft = torch.fft.fft(sequence_output, dim=-1)
        freq_len = x_fft.shape[-1]
        low_end = int(freq_len * 0.3)
        high_start = int(freq_len * 0.7)
        
        # Obtain amplitude and phase
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Generate low-frequency view
        high_mask = torch.ones_like(amplitude)
        if current_mask_ratio > 0:
            high_noise = torch.rand_like(amplitude[:, :, high_start:])
            high_mask[:, :, high_start:] = (high_noise >= current_mask_ratio).float()
        masked_amp_low = amplitude * high_mask
        
        # Generate high-frequency view
        low_mask = torch.ones_like(amplitude)
        if current_mask_ratio > 0:
            low_noise = torch.rand_like(amplitude[:, :, :low_end])
            low_mask[:, :, :low_end] = (low_noise >= current_mask_ratio).float()
        masked_amp_high = amplitude * low_mask
        
        # Reconstruct complex signal
        low_fft = masked_amp_low * torch.exp(1j * phase)
        high_fft = masked_amp_high * torch.exp(1j * phase)
        
        # Inverse Transformation
        low_view = torch.fft.ifft(low_fft, dim=-1).real
        high_view = torch.fft.ifft(high_fft, dim=-1).real
        
        return low_view, high_view


    def freq_mixup(self, seq_out1, seq_out2, alpha):
        
        if alpha.dim() == 4 and alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1) 
            
        f1 = torch.fft.fft(seq_out1, dim=-1)
        f2 = torch.fft.fft(seq_out2, dim=-1)
        
        amp1, amp2 = torch.abs(f1), torch.abs(f2)
        phase1, phase2 = torch.angle(f1), torch.angle(f2)
        mixed_amp = alpha * amp1 + (1 - alpha) * amp2
        dominant_phase = torch.where(alpha > 0.5, phase1, phase2)
        mixed_freq = mixed_amp * torch.exp(1j * dominant_phase)
        mixed_seq = torch.fft.ifft(mixed_freq, dim=-1).real
        
        return mixed_seq

    def get_band_contrast_mode(self, epoch):
        
        if not self.adaptive_band:
            return 'both'  
        
        if epoch < self.curriculum_epochs * 0.33:
            return 'low_only'    
        elif epoch < self.curriculum_epochs * 0.66:
            return 'high_only'   
        else:
            return 'cross'       
        
    def get_adaptive_temperature(self, epoch):
        
        base_tau = self.args.contrast_tau
        progress = min(1.0, epoch / self.curriculum_epochs)
        return base_tau * (2.0 - progress)

    def cross_entropy(self, seq_out, pos_ids, neg_ids, weight=None):
      
        pos_emb = self.model.item_embeddings(pos_ids)

        # Hard Negative Mining
        if self.use_hard_negative and self.model.training:
            hard_neg_ids = self.get_hard_negatives(seq_out, pos_ids, self.hn_candidates)
            if np.random.rand() < self.hard_negative_ratio:
                neg_emb = self.model.item_embeddings(hard_neg_ids)
            else:
                neg_emb = self.model.item_embeddings(neg_ids)
        else:
            neg_emb = self.model.item_embeddings(neg_ids)

        pos_logits = (seq_out * pos_emb).sum(dim=-1).view(-1)
        neg_logits = (seq_out * neg_emb).sum(dim=-1).view(-1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        if weight is not None:
            weight = weight.unsqueeze(-1)
            pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget * weight
            neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget * weight
            loss = torch.sum(pos_loss + neg_loss) / torch.sum(istarget)
        else:
            loss = torch.sum(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "valid"
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            
            adaptive_tau = self.get_adaptive_temperature(epoch)
            band_mode = self.get_band_contrast_mode(epoch)

            for i, (rec_batch, aug_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model.transformer_encoder(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                joint_loss = self.args.rec_weight * rec_loss

                # ---------- Frequency Contrastive Learning -------------#
                if epoch >= self.args.pretrain_epoch and self.use_freq_augment:
                    low_view, high_view = self.frequency_decomposition(sequence_output, epoch)
                    
                    if low_view is not None:
                        seq_rep = self.get_sequence_rep(sequence_output, input_ids)
                        low_rep = self.get_sequence_rep(low_view, input_ids)
                        high_rep = self.get_sequence_rep(high_view, input_ids)
                        
                        # Frequency-band Adaptive Contrast
                        contrast_losses = []
                        if band_mode == 'low_only':
                            contrast_losses.append(self.nt_xent_loss(seq_rep, low_rep, tau=adaptive_tau))
                        elif band_mode == 'high_only':
                            contrast_losses.append(self.nt_xent_loss(seq_rep, high_rep, tau=adaptive_tau))
                        elif band_mode == 'cross':
                            contrast_losses.append(self.nt_xent_loss(low_rep, high_rep, tau=adaptive_tau))
                        else: 
                            contrast_losses.append(self.nt_xent_loss(seq_rep, low_rep, tau=adaptive_tau))
                            contrast_losses.append(self.nt_xent_loss(seq_rep, high_rep, tau=adaptive_tau))
                            contrast_losses.append(self.nt_xent_loss(low_rep, high_rep, tau=adaptive_tau))
                        
                        freq_cl_loss = sum(contrast_losses) / len(contrast_losses)
                        
                        with torch.no_grad():
                            rec_mag = rec_loss.item()
                            cl_mag = freq_cl_loss.item()
                            scale_factor = (rec_mag / (cl_mag + 1e-8)) * 0.2
                            scale_factor = min(max(scale_factor, 0.1), 5.0)
                        
                        joint_loss += self.args.contrast_weight * scale_factor * freq_cl_loss

                # ---------- Single-sequence Augmentation -------------#
                if epoch >= self.args.pretrain_epoch:
                    for aug_batch in aug_batches:
                        aug_loss, _, _ = self.aug_mix_learning(input_ids, aug_batch, target_pos, target_neg)
                        joint_loss += self.args.aml_weight * aug_loss

                    # ---------- Cross-sequence Augmentation -------------#
                    if np.random.rand() < 0.5:  
                        freq_mix_loss = self.freq_whole_mix_learning(sequence_output, target_pos, target_neg, input_ids, epoch)
                        joint_loss += self.args.wml_weight * freq_mix_loss
                    else:
                        whole_mix_loss = self.whole_mix_learning(sequence_output, target_pos, target_neg, input_ids)
                        joint_loss += self.args.wml_weight * whole_mix_loss

                self.optim.zero_grad()
                joint_loss.backward()
                
                if self.use_hard_negative:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optim.step()
                rec_avg_loss += rec_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "band_mode": band_mode,
                "tau": '{:.4f}'.format(adaptive_tau)
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()
            rec_data_iter = tqdm(enumerate(dataloader),
                                 desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                 total=len(dataloader),
                                 bar_format="{l_bar}{r_bar}")
            pred_list = None
            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
    
    def freq_whole_mix_learning(self, seq_out, pos_ids, neg_ids, input_ids, epoch):
       
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        mix_rec_loss = 0
        
        for i in range(self.n_whole_level):
            alpha = self.beta_distribution.sample((seq_out.shape[0],)).to(self.device)
            if alpha.dim() == 2 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)
            alpha = alpha.view(seq_out.shape[0], 1, 1)
            
            indices = torch.randperm(seq_out.shape[0])
            mixed_seq_out = self.freq_mixup(seq_out, seq_out[indices], alpha)
            mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
            mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
            mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
            mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
            mix_rec_loss += torch.sum(
                - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)
            
        for i in range(self.n_whole_level):
            alpha = self.beta_distribution.sample((seq_out.shape[0], seq_out.shape[2])).to(self.device)
            if alpha.dim() == 3 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)
            alpha = alpha.view(seq_out.shape[0], 1, seq_out.shape[2])
            
            indices = torch.randperm(seq_out.shape[0])
            mixed_seq_out = self.freq_mixup(seq_out, seq_out[indices], alpha)
            mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
            mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
            mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
            mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
            mix_rec_loss += torch.sum(
                - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        return mix_rec_loss

    def whole_mix_learning(self, seq_out, pos_ids, neg_ids, input_ids):
        
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        mix_rec_loss = 0

        for i in range(self.n_whole_level):
            alpha = self.beta_distribution.sample((seq_out.shape[0], seq_out.shape[1])).to(self.device)
            indices = torch.randperm(seq_out.shape[0])

            mixed_seq_out = alpha * seq_out + (1 - alpha) * seq_out[indices]
            mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
            mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
            mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
            mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
            mix_rec_loss += torch.sum(
                - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        for i in range(self.n_whole_level):
            alpha = self.beta_distribution.sample((seq_out.shape[0], seq_out.shape[2])).permute(0, 2, 1).to(self.device)
            indices = torch.randperm(seq_out.shape[0])

            mixed_seq_out = alpha * seq_out + (1 - alpha) * seq_out[indices]
            mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
            mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
            mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
            mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
            mix_rec_loss += torch.sum(
                - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        return mix_rec_loss

    def aug_mix_learning(self, input_ids, aug_inputs, target_pos, target_neg):
    
        aug_loss = 0
        for aug_batch in aug_inputs:
            aug_sequences, aug_weight = aug_batch
            aug_weight = aug_weight.to(self.device)
            aug_sequences = aug_sequences.to(self.device)
            mix_weight = self.beta_distribution.sample(torch.tensor([input_ids.shape[0]])).unsqueeze(-1).to(self.device)

            loss_weight = 1.0 / (mix_weight.squeeze(-1).squeeze(-1) * aug_weight)
            loss_weight = (loss_weight - loss_weight.min()) / (loss_weight.max() - loss_weight.min())

            augment_output = self.model.transformer_encoder(input_ids, aug_sequences, mix_weight)
            aug_loss += self.cross_entropy(augment_output, target_pos, target_neg, loss_weight)
            
        return aug_loss, None, None

    def get_sequence_rep(self, sequence_output, input_ids):
        
        batch_size = sequence_output.size(0)
        non_pad_mask = (input_ids != 0).float()
        last_non_pad_idx = non_pad_mask.sum(dim=1).long() - 1
        last_non_pad_idx = torch.clamp(last_non_pad_idx, min=0)
        sequence_rep = sequence_output[torch.arange(batch_size), last_non_pad_idx, :]
        return sequence_rep

    def nt_xent_loss(self, rep1, rep2, tau=0.07):
        
        device = rep1.device
        N = rep1.size(0)
        reps = torch.cat([rep1, rep2], dim=0)
        sim_matrix = torch.nn.functional.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / tau
        pos_mask = torch.zeros((2 * N, 2 * N), dtype=torch.bool, device=device)
        pos_mask[:N, N:] = torch.eye(N, device=device, dtype=torch.bool)
        pos_mask[N:, :N] = torch.eye(N, device=device, dtype=torch.bool)
        neg_mask = ~torch.eye(2 * N, dtype=torch.bool, device=device) & ~pos_mask
        pos_logits = sim_matrix[pos_mask].view(2 * N, 1)
        neg_logits = sim_matrix[neg_mask].view(2 * N, -1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(2 * N, dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss