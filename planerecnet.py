"""Part of the code is adapted from:
    @dbolya yolact: https://github.com/aim-uofa/AdelaiDet/adet/modeling/solov2/solov2.py
    Licensed under The MIT License [see LICENSE for details]
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg
from utils import timer
from models.functions.nms import point_nms, matrix_nms, mask_nms
from models.functions.funcs import bias_init_with_prob
from models.fpn import FPN
from models.backbone import construct_backbone
from data.augmentations import FastBaseTransform


device = torch.device(cfg.device)

class PlaneRecNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.device)
        self.depth_decoder_indices = cfg.depth.selected_layers # use for shape detection
        self.fpn_indices = cfg.fpn.selected_layers

        # Instance parameters.
        self.num_classes = cfg.num_classes
        self.num_kernels = cfg.solov2.num_kernels
        self.num_grids = cfg.solov2.num_grids

        self.instance_in_features = cfg.solov2.instance_in_features
        self.instance_strides = cfg.solov2.fpn_instance_strides
        self.instance_in_channels = cfg.fpn.num_features  # = fpn.
        self.instance_channels = cfg.solov2.instance_channels

        # Mask parameters.
        self.mask_in_features = cfg.solov2.masks_in_features
        self.mask_in_channels = cfg.fpn.num_features
        self.mask_channels = cfg.solov2.masks_channels
        self.num_masks = cfg.solov2.num_masks

        # Inference parameters.
        self.max_before_nms = cfg.solov2.nms_pre
        self.score_threshold = cfg.solov2.score_thr
        self.update_threshold = cfg.solov2.update_thr
        self.mask_threshold = cfg.solov2.mask_thr
        self.max_per_img = cfg.solov2.top_k
        self.nms_kernel = cfg.solov2.nms_kernel
        self.nms_sigma = cfg.solov2.nms_sigma
        self.nms_type = cfg.solov2.nms_type
        
        # build backbone and fpn
        self.backbone = construct_backbone(cfg.backbone)
        if cfg.freeze_bn:
            self.freeze_bn()
        src_channels = self.backbone.channels
        self.fpn = FPN([src_channels[i] for i in self.fpn_indices], start_level=cfg.fpn.start_level)

        # build the ins head.
        instance_shapes = [cfg.fpn.num_features for _ in range(len(cfg.solov2.instance_in_features))]
        self.inst_head = SOLOv2InsHead(cfg, instance_shapes)

        # build the mask head.
        mask_shapes = [cfg.fpn.num_features for _ in range(len(cfg.solov2.masks_in_features))]
        self.mask_head = SOLOv2MaskHead(cfg, mask_shapes)

    
    def forward(self, x):

        # Backbone
        with timer.env("backbone"):
            features_encoder = self.backbone(x)
            #for i in features: print(i.shape)

        # Feature Pyramid Network
        with timer.env("fpn"):
            features = self.fpn([features_encoder[i] for i in self.fpn_indices])
        
        # Instance Branch
        with timer.env("instance head"):
            ins_features = [features[f] for f in range(len(self.instance_in_features))]
            ins_features = self.split_feats(ins_features)
            cate_pred, kernel_pred = self.inst_head(ins_features)
        
        # Mask Branch
        with timer.env('mask head'):
            mask_features = [features[f] for f in range(len(self.mask_in_features))]
            mask_pred = self.mask_head(mask_features)

        # Inference or output for trainng
        with timer.env('Inferencing'):
            if self.training:
                #mask_feat_size = mask_pred.size()[-2:]
                return mask_pred, cate_pred, kernel_pred
            else:
                # point nms.
                cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                            for cate_p in cate_pred]
                # do inference for results.
                results = self.inference(mask_pred, cate_pred, kernel_pred, x)

                return results
    
    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False),
                feats[1],
                feats[2],
                feats[3])
        
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        for name, module in self.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)  # or is_script_conv
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    if 'inst_head' in name and 'cate_pred' in name:
                        prior_prob = cfg.solov2.focal_loss_init_pi
                        bias_value = bias_init_with_prob(prior_prob)
                        module.bias.data.fill_(bias_value)
                    else:
                        module.bias.data.fill_(0)
                    
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def inference(self, pred_masks, pred_cates, pred_kernels, batched_images):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(batched_images)):
            # image size.
            ori_img = batched_images[img_idx]
            height, width = ori_img.size()[1], ori_img.size()[2]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                            for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)


            # pred_edge = pred_edges[img_idx, ...].unsqueeze(0)
            # inference for single image.
            result = self.inference_single_image(pred_mask, pred_cate, pred_kernel, ori_size)
            results.append(result)
        return results

    def inference_single_image(self, seg_preds, cate_preds, kernel_preds, ori_size):
        result = {'pred_masks': None, 'pred_boxes': None, 'pred_classes': None, 'pred_scores': None}

        # process.
        inds = (cate_preds > self.score_threshold)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return result

        # cate_labels & kernel_preds
        inds = inds.nonzero(as_tuple=False)
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # check gradient
        # import os
        # import numpy as np
        # for i in range(seg_preds.shape[0]):
        #     current_tensor = seg_preds[i, :, :].detach().cpu().numpy()
        #     current_tensor = ((current_tensor - current_tensor.min()) / (current_tensor.max() - current_tensor.min()) * 255).astype(np.uint8)
        #     # current_tensor = cv2.Canny(current_tensor,50,100, 1)
        #     tensor_color = cv2.applyColorMap(current_tensor, cv2.COLORMAP_VIRIDIS)
        #     tensor_color_path = os.path.join('seg_preds_gradient', '{}.png'.format(i))
        #     cv2.imwrite(tensor_color_path, tensor_color)

        # mask.
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return result

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        if self.nms_type == "matrix":
            # matrix nms & filter.
            cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                          sigma=self.nms_sigma, kernel=self.nms_kernel)
            keep = cate_scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                 nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError

        if keep.sum() == 0:
            return result

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        seg_masks = F.interpolate(seg_preds.unsqueeze(0),
                                  size=ori_size,
                                  mode='bilinear', align_corners=False).squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        result['pred_scores'] = cate_scores
        result['pred_classes'] = cate_labels
        result['pred_masks'] = seg_masks

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)
        for i in range(seg_masks.size(0)):
            mask = seg_masks[i].squeeze()
            ys, xs = torch.where(mask)
            pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        result['pred_boxes'] = pred_boxes     

        return result


class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        SOLOv2 Instance Head.
        """

        super().__init__()
        self.num_classes = cfg.num_classes
        self.num_kernels = cfg.solov2.num_kernels
        self.num_grids = cfg.solov2.num_grids
        self.instance_in_features = cfg.solov2.instance_in_features 
        self.instance_strides = cfg.solov2.fpn_instance_strides 
        self.instance_in_channels = cfg.fpn.num_features
        self.instance_channels = cfg.solov2.instance_channels 
        self.num_levels = len(self.instance_in_features) 
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        
        head_configs = {"cate": (cfg.solov2.num_instance_convs, 
                                 cfg.solov2.use_dcn_in_instance, 
                                 False),
                        "kernel": (cfg.solov2.num_instance_convs,
                                   cfg.solov2.use_dcn_in_instance, 
                                   cfg.solov2.use_coord_conv) 
                        }

        norm = None if cfg.solov2.norm == "none" else cfg.solov2.norm

        for head in head_configs:
            tower = []
            head_depth, use_deformable, use_coord = head_configs[head]
            for i in range(head_depth):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                        chn, self.instance_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            pass
        """
        cate_pred = []
        kernel_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=False)
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))
            
            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))
        return cate_pred, kernel_pred


class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg, input_shape):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        self.num_masks = cfg.solov2.num_masks
        self.mask_in_features = cfg.solov2.masks_in_features
        self.mask_in_channels = cfg.fpn.num_features
        self.mask_channels = cfg.solov2.masks_channels
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        norm = None if cfg.solov2.norm == "none" else cfg.solov2.norm

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            mask_pred (Tensor [C x W x H]): Fused mask prediciton. 
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            # feature_add_all_level += self.convs_all_levels[i](mask_feat)  # This inplace operation may cause RuntimeError for pytorch >= 1.10
            feature_add_all_level = feature_add_all_level.clone() + self.convs_all_levels[i](mask_feat) 

        mask_pred = self.conv_pred(feature_add_all_level)
        return mask_pred


if __name__ == "__main__":
    import argparse
    def parse_args(argv=None):
        parser = argparse.ArgumentParser(description="For PlaneRecNet Debugging and Inference Time Measurement")
        parser.add_argument(
            "--trained_model",
            default=None,
            type=str,
            help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.',
        )
        parser.add_argument(
            "--config", 
            default="PlaneRecNet_50_config", 
            help="The config object to use.")
        parser.add_argument(
            "--fps", 
            action="store_true", 
            help="Testing running speed.")
        global args
        args = parser.parse_args(argv)
    

    parse_args()
    from data.config import set_cfg
    from utils.utils import MovingAverage, init_console

    init_console()

    set_cfg(args.config)
    net = PlaneRecNet(cfg)
    if args.trained_model is not None:
        net.load_weights(args.trained_model)
    else:
        net.init_weights(backbone_path="weights/" + cfg.backbone.path)
        print(cfg.backbone.name)
        
    net.eval()
    net = net.cuda(0)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    frame = torch.from_numpy(cv2.imread("data/example_nyu.jpg", cv2.IMREAD_COLOR)).cuda(0).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    y = net(batch)
    

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    if args.fps:
        net(batch)
        avg = MovingAverage()
        try:
            while True:
                timer.reset()
                with timer.env("everything else"):
                    net(batch)
                avg.add(timer.total_time())
                print("\033[2J")  # Moves console cursor to 0,0
                timer.print_stats()
                print(
                    "Avg fps: %.2f\tAvg ms: %.2f         "
                    % (1000 / avg.get_avg(), avg.get_avg())
                )
        except KeyboardInterrupt:
            pass
    else:
        exit()
