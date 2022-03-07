# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 11:46
# @Author  : zxf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits

from utils.focal_loss import FocalLoss
from utils.label_smoothing import LabelSmoothingCrossEntropy


class BertSpanForNer(nn.Module):
    def __init__(self, model_name_or_path,
                 num_labels, loss_type,
                 soft_label):
        super(BertSpanForNer, self).__init__()
        self.soft_label = soft_label
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.hidden_dropout_prob = self.bert.config.hidden_dropout_prob
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(self.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(self.hidden_size + self.num_labels,
                                          self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(self.hidden_size + 1, self.num_labels)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, start_positions=None,
                end_positions=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.contiguous().view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.contiguous().view(-1)[active_loss]
            active_end_labels = end_positions.contiguous().view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs

