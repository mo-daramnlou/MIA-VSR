import torch
from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel

from archs import build_network
from .sr_model import SRModel1
from .video_recurrent_model import VideoRecurrentModel1
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from losses.sparsity_loss import SparsityLoss
from basicsr.utils import get_root_logger, get_time_str, imwrite, tensor2img
import os
import os.path as osp

@MODEL_REGISTRY.register()
class RecurrentMixPrecisionRTModel(VideoRecurrentModel1):
    """VRT Model adopted in the original VRT. Mix precision is adopted.

    Paper: A Video Restoration Transformer
    """

    def __init__(self, opt):
        super(SRModel1, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.net_g.to(self.device)
        self.print_network(self.net_g)


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', False), param_key) #True

        if self.is_train:
            self.init_training_settings()
            self.fix_flow_iter = opt['train'].get('fix_flow')
            self.kd_enabled = opt['train']['knowledge_distillation'].get('enabled')
            self.kd_start_iter= opt['train']['knowledge_distillation'].get('start_iter')
        self.cri_spa = SparsityLoss(opt['train'].get('parsity_target'))
        self.embed_dim = opt['network_g'].get('embed_dim')
        self.ff = True


    # add use_static_graph
    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
            use_static_graph = self.opt.get('use_static_graph', False)
            if use_static_graph:
                logger = get_root_logger()
                logger.info(
                    f'Using static graph. Make sure that "unused parameters" will not change during training loop.')
                net._set_static_graph()
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                # add 'deform'
                if 'spynet' in name or 'deform' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])

        # # adopt mix precision
        # use_apex_amp = self.opt.get('apex_amp', True)
        # if use_apex_amp:
        #     self.net_g, self.optimizer_g = amp.initialize(
        #         self.net_g, self.optimizer_g, init_args=dict(opt_level='O1'))
        #     logger = get_root_logger()
        #     logger.info(f'Using apex mix precision to accelerate.')

        # adopt DDP
        self.net_g = self.model_to_device(self.net_g)
        self.optimizers.append(self.optimizer_g)


    def compute_multi_anchor_kd_loss(self, anchor_feats, anchor_gt, kd_weight=1.0):
        """
        Compute multi-anchor knowledge distillation loss for the four modules.
        Args:
            anchor_feats (dict): Student features for each anchor/module.
            anchor_gt (dict): Teacher features for each anchor/module.
            kd_weight (float): Weight for the KD loss.
        Returns:
            torch.Tensor: Total KD loss.
        """
        kd_loss = 0.0
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for module in modules:
            if module in anchor_feats and module in anchor_gt:
                student_feats = anchor_feats[module].permute(1,0,2,3,4).squeeze(0)
                teacher_feats = anchor_gt[module].squeeze(0)
                teacher_feats = teacher_feats * (self.embed_dim/120) # 120: MIA_FULL embed dim
                # print("types: ",type(student_feats), type(teacher_feats))
                # print("shapes: ",student_feats.shape, teacher_feats.shape)
                kd_loss = kd_loss + self.cri_pix(student_feats, teacher_feats)
                # Assume both are lists of tensors (one per frame)
                # for s_feat, t_feat in zip(student_feats, teacher_feats):
                #     kd_loss = kd_loss + torch.nn.functional.mse_loss(s_feat, t_feat)
        return 0.0001 * 3 * kd_weight * kd_loss
    


    # def normalize_tensor(self,tensor):
    #     min_val = tensor.min()
    #     max_val = tensor.max()
    #     normalized_tensor = (tensor - min_val) / (max_val - min_val)
        # return normalized_tensor

    def optimize_parameters(self, scaler, current_iter):
        
        # print("optimize_parameters gt:", self.gt.shape)
        # print("optimize_parameters lq:", self.lq.shape)
        # print("optimize_parameters b1:", self.anchor_gt_b1.shape)
        # print("optimize_parameters f1:", self.anchor_gt_f1.shape)
        # print("optimize_parameters b2:", self.anchor_gt_b2.shape)
        # print("optimize_parameters f2:", self.anchor_gt_f2.shape)

        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'deform' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        # update the gradient when forward 4 times
        self.optimizer_g.zero_grad()

        with autocast():
            self.output, masks, anchor_feats = self.net_g(self.lq)

            # if self.ff:
            #     self.ff = False
            #     save_folder = f'/home/mohammad/Documents/uni/deep learning/FinalProject/inference_data/kd_train3/'
            #     o = {
            #         'lq': self.lq,
            #         'b1': self.anchor_gt_b1,
            #         'f1': self.anchor_gt_f1,
            #         'b2': self.anchor_gt_b2,
            #         'f2': self.anchor_gt_f2,
            #         'gt': self.gt,
            #         'out': self.output
            #     }
            #     for i in range(0,8):
            #         for key in o:
            #             o_current = o[key].reshape(-1, *o[key].shape[2:])
            #             print(f'o_current {key}', o_current)
            #             print(f'{i}_{key}', o_current[i].shape)
            #             feat = tensor2img(self.normalize_tensor(o_current[i].squeeze()), rgb2bgr=True, min_max=(0,1))
            #             s_folder = osp.join(save_folder, f'{i}_{key}' + '.png')
            #             imwrite(feat, s_folder)

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
            # sparsity loss
            if masks is not None:
                l_spa = self.cri_spa(masks)
                l_total += l_spa
                loss_dict['l_spa'] = l_spa
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

            # Multi-anchor knowledge distillation loss
            if self.kd_enabled and current_iter > self.kd_start_iter and self.anchor_gt_b1 is not None and self.anchor_gt_f1 is not None and self.anchor_gt_b2 is not None and self.anchor_gt_f2 is not None:
                anchor_gt = {'backward_1': self.anchor_gt_b1, 'forward_1': self.anchor_gt_f1, 'backward_2': self.anchor_gt_b2, 'forward_2': self.anchor_gt_f2}
                kd_weight = getattr(self, 'kd_weight', 1.0)
                l_kd = self.compute_multi_anchor_kd_loss(anchor_feats, anchor_gt, kd_weight)
                l_total += l_kd
                loss_dict['l_kd'] = l_kd

            scaler.scale(l_total).backward()
            scaler.step(self.optimizer_g)
            scaler.update()

            # l_total.backward()
            # self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


