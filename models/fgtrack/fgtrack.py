"""
Basic fgtrack model.
"""
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.position_encoding import build_position_encoding
from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.head import build_box_head
from lib.models.fgtrack.vit import vit_base_patch16_224
from lib.models.fgtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import numpy as np
from lib.utils.box_ops import box_cxcywh_to_xyxy
from lib.models.fgtrack.HistoricalPromptNetwork import HistoricalPromptNetwork
from lib.models.fgtrack.modules import KeyProjection, ResBlock
from sklearn.cluster import estimate_bandwidth
from GridShiftPP2 import GridShiftPP

class FGTrack(nn.Module):
    """ This is the base class for fgtrack """

    def __init__(self, transformer, box_head, transformer_dec, position_encoding,aux_loss=False, head_type="CORNER", vis_during_train=False, new_hip=False, memory_max=150, update_interval=20):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.HIP = HistoricalPromptNetwork()
        self.key_proj = KeyProjection(768, keydim=64)
        self.key_comp = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.searchRegionFusion = ResBlock(768, 768)
        self.new_hip = new_hip
        self.update_interval = update_interval
        if self.new_hip:
            self.upsample = nn.Upsample(scale_factor=2.0, align_corners=True, mode="bilinear")
        self.memorys = []
        self.mem_max = memory_max

        self.transformer_dec = transformer_dec
        self.position_encoding = position_encoding
        self.query_embed=nn.Embedding(num_embeddings=1, embedding_dim=768)

    def set_eval(self):
        self.HIP.set_eval(mem_max=self.mem_max)

    def forward(self, template: torch.Tensor,  #(bs,3,128,128)
                search: list,  #5个元素的list 每个元素为(bs,3,256,256)
                search_after: torch.Tensor=None,
                previous: torch.Tensor=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gtBoxes=None,
                previousBoxes=None,
                template_boxes=None,
                training=True,  # True
                tgt_pre=None,
                ):
        '''
            template : [B 3 H_z W_z]
            search : [3 * [B 3 H_x W_x]]
            previous : [B L 3 H_x W_x]
        '''
        self.ce_template_mask=ce_template_mask
        num_search = len(search)
        B, _, Ht, Wt = template.shape
        _, _, Hs, Ws = search[0].shape

        x, aux_dict = self.backbone(z=template, x=search[0],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)#(bs,320,768)
        _, _, C = x.shape

        x_decs = []
        query_embed = self.query_embed.weight  # (1,768)
        assert len(query_embed.size()) in [2, 3]
        if len(query_embed.size()) == 2:
            query_embeding = query_embed.unsqueeze(1)

        tgt_all = [torch.zeros_like(query_embeding.expand(-1,B,-1)) for _ in range(num_search + 1)]
        # template_mask
        upsampled_template = self.upsample(template)  # (1,3,256,256)
        template_mask = self.generateMask([None, None, None],
                                          template_boxes.squeeze(0),
                                          upsampled_template, x,
                                          visualizeMask=False, cxcywh=False, seqName=0, frame=0)
        template_feature = x[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        template_feature = self.upsample(template_feature)  # (1,768,16,16)
        ref_v_template = self.HIP('encode',
                                  upsampled_template,
                                  template_feature,
                                  template_mask.unsqueeze(1))  # ref_v_template (1,384,1,16,16)
        ref_v_template = ref_v_template.view(B, C, 1, -1).squeeze(-2)  # C 768 #(B,768,256)

        pos_embed = self.position_encoding(B)#(1,B,768)
        tgt_q = tgt_all[0]
        tgt_kv = torch.cat(tgt_all[:1], dim=0)
        if not training and len(tgt_pre) != 0:
            tgt_kv = torch.cat(tgt_pre, dim=0)
        tgt = [tgt_q, tgt_kv]  #

        tgt_out = self.transformer_dec(ref_v_template.permute(2, 0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
        x_decs.append(tgt_out[0])
        tgt_all[0] = tgt_out[0]  # 到此，这个batch的模板特征decoder结束
        out_list = []
        for i in range(num_search):  # 5次循环
            ###
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            out = self.forward_head(feat_last, tgt_all[i], None)
            out_list.append(out)
            ###
            if i == num_search - 1:
                break
            # search0_mask
            mask = self.generateMask(aux_dict['removed_indexes_s'],
                                         out['pred_boxes'].squeeze(1), search[i], x,
                                     visualizeMask=False, seqName=i+1,frame_idx=i+1,attn=aux_dict['attn'],global_index_s=aux_dict['global_index_s'])
            searchRegionFeature_1 = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(-1, C, Hs // 16, Ws // 16)

            ref_v = self.HIP('encode',
                             search[i],
                             searchRegionFeature_1,
                             mask.unsqueeze(1))
            ref_v = ref_v.view(B, C, 1, -1).squeeze(-2)  # C 768

            pos_embed = self.position_encoding(B)
            tgt_q = tgt_all[i + 1]
            tgt_kv = torch.cat(tgt_all[:i + 2], dim=0)
            if not training and len(tgt_pre) != 0:
                tgt_kv = torch.cat(tgt_pre, dim=0)
            tgt = [tgt_q, tgt_kv]  #
            tgt_out = self.transformer_dec(ref_v.permute(2, 0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
            x_decs.append(tgt_out[0])
            tgt_all[i + 1] = tgt_out[0]  #
            x, aux_dict = self.backbone(z=template, x=search[i+1],
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, previous_frames=previous,
                                            previous_anno=previousBoxes)

            if not training:
                if len(tgt_pre) < 5:
                    tgt_pre.append(tgt_out[0])
                else:
                    tgt_pre.pop(0)
                    tgt_pre.append(tgt_out[0])

        return out_list

    def forward_eval(self, template: torch.Tensor,  #(bs,3,128,128)
                search: list,  #5个元素的list 每个元素为(bs,3,256,256)
                search_after: torch.Tensor=None,
                previous: torch.Tensor=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gtBoxes=None,
                previousBoxes=None,
                template_boxes=None,
                index=None,
                training=True,  # True
                tgt_pre=None,
                info=None
                ):
        '''
            template : [B 3 H_z W_z]
            search : [3 * [B 3 H_x W_x]]
            previous : [B L 3 H_x W_x]
        '''
        self.ce_template_mask = ce_template_mask
        if index <= 10:
            x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            num_search, _, Hs, Ws = search.shape

            # template_mask
            upsampled_template = self.upsample(template)  # (1,3,256,256)
            template_mask = self.generateMask([None, None, None],
                                              template_boxes,
                                              upsampled_template, x,
                                              visualizeMask=False, cxcywh=False, seqName=0, frame=0)
            template_feature = x[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
            template_feature = self.upsample(template_feature)  # (1,768,16,16)
            ref_v_template = self.HIP('encode',
                                      upsampled_template,
                                      template_feature,
                                      template_mask.unsqueeze(1))  # ref_v_template (1,384,1,16,16)
            ref_v_template = ref_v_template.view(B, C, 1, -1).squeeze(-2)  # C 768 #(B,768,256)

            x_decs = []
            query_embed = self.query_embed.weight  # (1,768)
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                query_embeding = query_embed.unsqueeze(1)

            tgt_all = [torch.zeros_like(query_embeding.expand(-1,B,-1)) for _ in range(num_search)]


            pos_embed = self.position_encoding(B)#(1,B,768)
            tgt_q = tgt_all[0]
            tgt_kv = torch.cat(tgt_all[:1], dim=0)
            if not training and len(tgt_pre) != 0:
                tgt_kv = torch.cat(tgt_pre, dim=0)
            tgt = [tgt_q, tgt_kv]  #

            tgt_out = self.transformer_dec(ref_v_template.permute(2, 0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
            x_decs.append(tgt_out[0])
            tgt_all[0] = tgt_out[0]  # 到此，这个batch的模板特征decoder结束

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            out = self.forward_head(feat_last, tgt_all[0], None)
            out.update(aux_dict)
            out['tgt'] = tgt_pre

            if index == 10:
                mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search, x,
                                         visualizeMask=False, frame=index, seqName=info,attn=aux_dict['attn'],global_index_s=aux_dict['global_index_s'])

                searchRegionFeature_1 = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(-1, C, Hs // 16, Ws // 16)

                ref_v = self.HIP('encode',
                                 search,
                                 searchRegionFeature_1,
                                 mask.unsqueeze(1))
                ref_v = ref_v.view(B, C, 1, -1).squeeze(-2)  # C 768

                pos_embed = self.position_encoding(B)
                tgt_all = [torch.zeros_like(query_embeding.expand(-1, B, -1)) for _ in range(num_search)]
                tgt_q = tgt_all[0]
                tgt_kv = torch.cat(tgt_all[:1], dim=0)
                if not training and len(tgt_pre) != 0:
                    tgt_kv = torch.cat(tgt_pre, dim=0)
                tgt = [tgt_q, tgt_kv]  #
                tgt_out = self.transformer_dec(ref_v.permute(2, 0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
                x_decs.append(tgt_out[0])
                tgt_all[0] = tgt_out[0]  #


                if len(tgt_pre) < 5: #num_search-1
                    tgt_pre.append(tgt_out[0])
                else:
                    tgt_pre.pop(0)
                    tgt_pre.append(tgt_out[0])
            out['tgt'] = tgt_pre
            return out
        else:
            x, aux_dict = self.backbone(z=template, x=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape


            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            out = self.forward_head(feat_last, tgt_pre[-1], None)

            out.update(aux_dict)

            mask = self.generateMask(aux_dict['removed_indexes_s'],
                                     out['pred_boxes'].squeeze(1),
                                     search, x, visualizeMask=False, frame=index, seqName=info,attn=aux_dict['attn'],global_index_s=aux_dict['global_index_s'])

            searchRegionFeature = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(-1, C, Hs // 16, Ws // 16)

            ref_v = self.HIP('encode',
                             search,
                             searchRegionFeature,
                             mask.unsqueeze(1))

            ref_v = ref_v.view(B, C, 1, -1).squeeze(-2)  # C 768

            pos_embed = self.position_encoding(B)
            query_embed = self.query_embed.weight  # (1,768)
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                query_embeding = query_embed.unsqueeze(1)
            tgt_all = [torch.zeros_like(query_embeding.expand(-1, B, -1)) for _ in range(B)]
            tgt_q = tgt_all[0]
            tgt_kv = torch.cat(tgt_all[:1], dim=0)
            if not training and len(tgt_pre) != 0:
                tgt_kv = torch.cat(tgt_pre, dim=0)
            tgt = [tgt_q, tgt_kv]  #
            tgt_out = self.transformer_dec(ref_v.permute(2, 0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
            tgt_all[0] = tgt_out[0]  #

            if len(tgt_pre) < 5:  # num_search-1
                tgt_pre.append(tgt_out[0])
            else:
                tgt_pre.pop(0)
                tgt_pre.append(tgt_out[0])

            out['tgt'] = tgt_pre

            return out


    def deNorm(self, image):
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        return img

    def generateMask(self, ceMasks, predBoxes, img_normed, img_feat, visualizeMask=False, cxcywh=True, frame=None, seqName=None,frame_idx=None,attn=None,global_index_s=None):
        B, _, H_origin, W_origin = img_normed.shape
        masks = torch.zeros((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        pure_ce_masks = torch.ones((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        stride = 16
        h_num = H_origin // stride
        box_list = []
        # visualizeMask = True
        for i in range(B):
            if cxcywh:
                box = (box_cxcywh_to_xyxy((predBoxes[i])) * H_origin).int()
            else:
                box = (predBoxes[i] * H_origin).int()
                box[2] += box[0]
                box[3] += box[1]

            box[0] = 0 if box[0] < 0 else box[0]
            box[1] = H_origin if box[1] > H_origin else box[1]
            box[2] = W_origin if box[2] > W_origin else box[2]
            box[3] = 0 if box[3] < 0 else box[3]

            box_list.append([box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()])
            # visualizeMask=True
            if visualizeMask:
                if not os.path.exists(f"./masks_vis/{seqName}/{i}"):
                    os.makedirs(f"./masks_vis/{seqName}/{i}")
                img = self.deNorm(img_normed[i])
            #masks[i] = torch.zeros((H_origin, W_origin), dtype=np.uint8)
            masks[i][box[1].item():box[3].item(), box[0].item():box[2].item()] = 1
            if ceMasks[0] is not None and ceMasks[1] is not None and ceMasks[2] is not None:
                ce1 = ceMasks[0][i]
                ce2 = ceMasks[1][i]
                ce3 = ceMasks[2][i]

                ce = torch.cat([ce1, ce2, ce3], axis=0)
                for num in ce:
                    x = int(num) // h_num
                    y = int(num) % h_num
                    masks[i][x*16 : (x+1)*16, y*16 : (y+1)*16] = 0
                    pure_ce_masks[i][x*16 : (x+1)*16, y*16 : (y+1)*16] = 0

        if attn is not None:
            # attn_t = attn[:, :, :64, 64:]  # attention scores of [template-to-search region]
            attn_t = attn[:, :, :144, 144:]  # attention scores of [template-to-search region]
            bs = attn_t.shape[0]
            if self.ce_template_mask is not None:
                box_mask_z = self.ce_template_mask.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1,
                                                                                     attn_t.shape[-1])
                attn_t = attn_t[box_mask_z]
                attn_t = attn_t.view(bs, 12, -1, box_mask_z.shape[-1])
                attn_t = attn_t.mean(dim=2).mean(dim=1)
            if ceMasks[0] is not None and ceMasks[1] is not None and ceMasks[2] is not None:
                removed_indexes_cat = torch.cat(ceMasks, dim=1)  # (bs,3) attn(bs,12,319,319)
                attn_t = self.restore_vector(attn_t, global_index_s, removed_indexes_cat)
            att_masks = self.attention_mask_GridShift(attn_t,box_list,device=img_feat.device)

            att_masks = self.constraint_mask_batch(att_masks, box_list)
            masks += att_masks.byte()

        return masks

    def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
        B, HW, C = cat_feature[:, -self.feat_len_s:].shape
        H = int(HW ** 0.5)
        W = H
        originSearch = cat_feature[:, -self.feat_len_s:].view(B, H, W, C).permute(0, 3, 1, 2)#(1,256,768)

        enc_opt = cat_feature[:, -self.feat_len_s:]
        dec_opt = out_dec.transpose(0,1).transpose(1,2)
        att = torch.matmul(enc_opt, dec_opt)
        aqa_out = enc_opt.unsqueeze(-1) * att.unsqueeze(-2)
        dynamicSearch = aqa_out.squeeze(-1).view(B, H, W, C).permute(0, 3, 1, 2)
        enc_opt = self.searchRegionFusion(originSearch + dynamicSearch).view(B, C, HW).permute(0, 2, 1)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        #Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map,_ = self.box_head(opt_feat, None)

            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def attention_mask_GridShift(self, attn_t,box_list, device=None):  # 256->(bs,12,320,320)
        attn_t = attn_t.reshape(-1, 24, 24)
        attn_t_np = attn_t.cpu().detach().numpy()
        batch = attn_t_np.shape[0]
        res = []
        for i in range(batch):
            attn = attn_t_np[i]
            pixels = attn.reshape(-1, 1)  # Reshape for clustering
            bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)
            grid_shift_instance = GridShiftPP(bandwidth=bandwidth,threshold=0.0001, iterations=10)
            label_map,cluster_centers = grid_shift_instance.fit_predict(pixels)
            labels = label_map.reshape(24, 24)  # Reshape labels back to the original shape

            # Find the most common cluster
            main_cluster = np.bincount(labels.flat).argmax()
            binary_mask = (labels != main_cluster).astype(int)
            res.append(binary_mask)

        res_np = np.array(res)
        res_torch = torch.tensor(res_np, device=device, dtype=torch.uint8)
        import torch.nn.functional as F
        upscaled_mask = F.interpolate(res_torch.float().unsqueeze(1), size=(384,384), mode='nearest').squeeze(1)
        return upscaled_mask

    def constraint_mask_batch(self, masks, bboxes):
        """
        Apply constraints to a batch of masks based on corresponding bounding boxes.

        masks: numpy array of shape (batch_size, H, W)
        bboxes: list of lists, each sublist is [x1, y1, w, h] for each bounding box in the batch

        Returns:
        numpy array of shape (batch_size, H, W) with constraints applied
        """
        batch_size = masks.shape[0]
        constrained_masks = torch.zeros_like(masks)  # 使用 torch.zeros_like 来创建同样形状的全零张量

        for i in range(batch_size):
            mask = masks[i]
            bbox = bboxes[i]

            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[0] + bbox[2])
            y2 = int(bbox[1] + bbox[3])

            # Apply constraints
            constrained_mask = mask.clone()  # 使用 clone 来复制张量
            constrained_mask[0:y1 + 1, :] = 0
            constrained_mask[y2:, :] = 0
            constrained_mask[:, 0:x1 + 1] = 0
            constrained_mask[:, x2:] = 0

            constrained_masks[i] = constrained_mask

        return constrained_masks

    def restore_vector(self, raw_attention, global_index_s, idx_remove):
        # batch_size, original_size, new_size = raw_attention.shape[0], raw_attention.shape[1], 256
        batch_size, original_size, new_size = raw_attention.shape[0], raw_attention.shape[1], 576
        pad_size = new_size - original_size
        pad_x = torch.zeros([batch_size, pad_size], device=raw_attention.device)
        attention = torch.cat([raw_attention, pad_x], dim=1)
        # 将global_index_s和idx_remove合并为完整的索引数组
        index_all = torch.cat([global_index_s, idx_remove], dim=1)

        # 使用scatter_填充extended_attention
        # 使用扩展索引和attention数据进行scatter_操作
        extended_attention = torch.zeros_like(attention).scatter_(dim=1, index=index_all.to(torch.int64), src=attention)

        return extended_attention
def build_fgtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('fgtrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    transformer_dec = build_transformer_dec(cfg, hidden_dim)
    position_encoding = build_position_encoding(cfg, sz = 1)

    box_head = build_box_head(cfg, hidden_dim)

    model = FGTrack(
        backbone,
        box_head,
        transformer_dec,
        position_encoding,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        new_hip=cfg.MODEL.NEW_HIP,
        memory_max=cfg.MODEL.MAX_MEM,
        update_interval=cfg.TEST.UPDATE_INTERVAL
    )

    if ('fgtrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models', cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model