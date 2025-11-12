from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch


class SeqTrackActor(BaseActor):
    """ Actor for training the SeqTrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.BINS = cfg.MODEL.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, target_seqs = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(outputs, target_seqs)

        return loss, status

    def forward_pass(self, data):
        n, b, _, _, _ = data['search_images'].shape   # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b,dim=0)
        feature_xz = self.net(images_list=template_list+search_list, mode='encoder') # forward the encoder

        bins = self.BINS # coorinate token
        start = bins + 1 # START token
        end = bins # End token
        len_embedding = bins + 2 # number of embeddings, including the coordinate tokens and the special tokens

        # box of search region
        targets = data['search_anno'].permute(1,0,2).reshape(-1, data['search_anno'].shape[2])   # x0y0wh
        targets = box_xywh_to_xyxy(targets)   # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets)) # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))

        # different formats of sequence, for ablation study
        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        box = (targets * (bins - 1)).int() # discretize the coordinates

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        # inpute sequence
        input_start = torch.ones([batch, 1]).to(box) * start
        input_seqs = torch.cat([input_start, box], dim=1)
        input_seqs = input_seqs.reshape(b,n,input_seqs.shape[-1])
        input_seqs = input_seqs.flatten(1)

        # target sequence
        target_end = torch.ones([batch, 1]).to(box) * end
        target_seqs = torch.cat([box, target_end], dim=1)
        target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1])
        target_seqs = target_seqs.flatten()
        target_seqs = target_seqs.type(dtype=torch.int64)

        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")

        outputs = outputs[-1].reshape(-1, len_embedding)

        return outputs, target_seqs

    def compute_losses(self, outputs, targets_seq, return_status=True):
        # Get loss
        ce_loss = self.objective['ce'](outputs, targets_seq)
        # weighted sum
        loss = self.loss_weight['ce'] * ce_loss

        # Stabilize logits before softmax (avoid NaN from inf logits)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6).softmax(-1)
        outputs = outputs[:, :self.BINS]
        value, extra_seq = outputs.topk(dim=-1, k=1)
        boxes_pred = extra_seq.squeeze(-1).reshape(-1,5)[:, 0:-1].float()
        # Clamp predicted discrete coords into valid range
        boxes_pred = boxes_pred.clamp(min=0.0, max=float(self.BINS - 1))
        boxes_target = targets_seq.reshape(-1,5)[:,0:-1].float()
        # Convert to xyxy
        boxes_pred_xyxy = box_cxcywh_to_xyxy(boxes_pred)
        boxes_target_xyxy = box_cxcywh_to_xyxy(boxes_target)
        # Ensure finite
        boxes_pred_xyxy = torch.nan_to_num(boxes_pred_xyxy, nan=0.0, posinf=0.0, neginf=0.0)
        boxes_target_xyxy = torch.nan_to_num(boxes_target_xyxy, nan=0.0, posinf=0.0, neginf=0.0)
        # Enforce proper ordering and minimal size
        x0p, y0p, x1p, y1p = boxes_pred_xyxy.unbind(-1)
        x0p_, x1p_ = torch.minimum(x0p, x1p), torch.maximum(x0p, x1p)
        y0p_, y1p_ = torch.minimum(y0p, y1p), torch.maximum(y0p, y1p)
        eps = 1e-6
        x1p_ = torch.maximum(x1p_, x0p_ + eps)
        y1p_ = torch.maximum(y1p_, y0p_ + eps)
        boxes_pred_xyxy = torch.stack([x0p_, y0p_, x1p_, y1p_], dim=-1)

        x0t, y0t, x1t, y1t = boxes_target_xyxy.unbind(-1)
        x0t_, x1t_ = torch.minimum(x0t, x1t), torch.maximum(x0t, x1t)
        y0t_, y1t_ = torch.minimum(y0t, y1t), torch.maximum(y0t, y1t)
        x1t_ = torch.maximum(x1t_, x0t_ + eps)
        y1t_ = torch.maximum(y1t_, y0t_ + eps)
        boxes_target_xyxy = torch.stack([x0t_, y0t_, x1t_, y1t_], dim=-1)
        # Compute IoU with robustness to degenerate boxes
        iou_vec, union_vec = box_iou(boxes_pred_xyxy, boxes_target_xyxy)
        valid = torch.isfinite(iou_vec) & (union_vec > eps)
        iou_value = None
        if valid.any():
            iou_value = iou_vec[valid].mean().item()

        if return_status:
            # status for log
            status = {"Loss/total": loss.item()}
            # Only update IoU meter if we computed a valid value; otherwise, keep previous average
            if iou_value is not None and torch.isfinite(torch.tensor(iou_value)):
                status["IoU"] = float(iou_value)
            return loss, status
        else:
            return loss

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)