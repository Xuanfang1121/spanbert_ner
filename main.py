# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 12:28
# @Author  : zxf
import os
import json

import torch
import numpy as np
from torch.optim import Adam
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from common.common import logger
from config.config import get_argparse
from utils.util import seed_everything
from utils.util import SpanEntityScore
from utils.util import bert_extract_item
from utils.util import save_file_to_json
from dataset.span_ner_dataset import collate_fn
from models.SpanBertNerModel import BertSpanForNer
from dataset.span_ner_dataset import SpannerProcessor
from dataset.span_ner_dataset import load_and_cache_examples


def evaluate(model, eval_features, id2label, device):
    metric = SpanEntityScore(id2label)
    # Eval!
    logger.info("  Num examples = %d", len(eval_features))
    eval_loss = 0.0
    nb_eval_steps = 0
    # todo 修改为batch
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(device)
        attention_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(device)
        start_positions = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(device)
        end_positions = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(device)
        subjects = f.subjects
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids, attention_mask,
                            start_positions, end_positions)

        tmp_eval_loss, start_logits, end_logits = outputs[:3]
        R = bert_extract_item(start_logits, end_logits)
        T = subjects
        metric.update(true_subject=T, pred_subject=R)
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    f1 = eval_info['f1']
    return f1


def train():
    args = get_argparse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = "cpu" if args.gpu_ids == "-1" else "cuda"
    # 设置随机种子数
    seed_everything(args.seed)
    # 数据处理类
    processor = SpannerProcessor()
    # 获取数据集的标签
    labels = processor.get_labels()
    labels2id = {label: i for i, label in enumerate(labels)}
    id2label = {value: key for key, value in labels2id.items()}
    save_file_to_json(labels2id, os.path.join(args.output_dir, args.label2id_file))
    logger.info("entity label size: {}".format(len(labels2id)))
    logger.info("save label finish")
    # 更新实体类型数量
    args.num_labels = len(labels)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)

    # 获取训练集特征
    train_dataset = load_and_cache_examples(args, processor, tokenizer, data_type='train')
    dev_dataset = load_and_cache_examples(args, processor, tokenizer, data_type='dev')
    logger.info("create train data feature and dev_dataset")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)
    print("train data batch num: ", len(train_dataloader))

    # model
    model = BertSpanForNer(args.model_name_or_path, args.num_labels,
                           args.loss_type, args.soft_label)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss = []
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data[0].to(device)
            attention_mask = batch_data[1].to(device)
            token_type_ids = batch_data[2].to(device)
            start_positions = batch_data[3].to(device)
            end_positions = batch_data[4].to(device)
            batch_data_len = batch_data[5]
            batch_outputs = model(input_ids, token_type_ids, attention_mask,
                                  start_positions, end_positions)
            batch_train_loss = batch_outputs[0]
            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()
            train_loss.append(batch_train_loss.tolist())
            if step % args.logging_steps == 0:
                logger.info("epoch: {}/{} step: {}/{} train loss:{}".format(epoch + 1,
                                                                            args.epochs,
                                                                            step + 1,
                                                                            len(train_dataloader),
                                                                            np.mean(train_loss)))
        logger.info("模型进行验证")
        f1 = evaluate(model, dev_dataset, id2label, device)
        if f1 >= best_f1:
            best_f1 = f1
            logger.info("model best f1 score: {}".format(best_f1))
            torch.save(model.state_dict(), os.path.join(args.output_dir,
                                                        args.save_model_name))


if __name__ == "__main__":
    train()