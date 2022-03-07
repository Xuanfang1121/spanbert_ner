# -*- coding: utf-8 -*-
# @Time    : 2022/1/9 15:26
# @Author  : zxf
import os
import json

import torch
from transformers import BertTokenizer

from common.common import logger
from config.config import get_argparse
from utils.util import load_json_file
from utils.util import bert_extract_item
from models.SpanBertNerModel import BertSpanForNer
from dataset.span_ner_dataset import SpannerProcessor
from dataset.span_ner_dataset import convert_examples_to_features_infer


"""
     加载模型做推理，这里是加载原生模型
"""


def predict():
    args = get_argparse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = "cpu" if args.gpu_ids == "-1" else "cuda"
    # read infer data
    test_data = []
    with open(os.path.join(args.data_dir, "test_demo.json"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = list(line["text"])
            test_data.append({"words": text})
    logger.info("test data size: {}".format(len(test_data)))
    # load label json file
    label2id = load_json_file(os.path.join(args.output_dir, args.label2id_file))
    id2label = {value: key for key, value in label2id.items()}
    args.num_labels = len(label2id)
    logger.info("entity label size: {}".format(len(label2id)))
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    # 数据处理类
    processor = SpannerProcessor()
    # get inder data example
    examples = processor._create_examples(test_data, "test")
    # get test data features
    test_features = convert_examples_to_features_infer(examples=examples,
                                                       max_seq_length=args.max_seq_length,
                                                       tokenizer=tokenizer,
                                                       cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                       pad_on_left=bool(args.model_type in ['xlnet']),
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       # pad on the left for xlnet
                                                       pad_token=tokenizer.convert_tokens_to_ids(
                                                           [tokenizer.pad_token])[0],
                                                       pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,)

    # model
    model = BertSpanForNer(args.model_name_or_path, args.num_labels,
                           args.loss_type, args.soft_label)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.save_model_name),
                                     map_location=torch.device(device)),
                          strict=True)
    model.to(device)
    model.eval()
    result = []
    for step, f in enumerate(test_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(device)
        attention_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids, attention_mask,
                            None, None)

        start_logits, end_logits = outputs
        R = bert_extract_item(start_logits, end_logits)
        print(R)
        decoder_entity = []
        if R:
            for item in R:
                decoder_entity.append({"tag": id2label[item[0]],
                                       "index": item[1],
                                       "name": ''.join(test_data[step]["words"][item[1]: item[2] + 1])})
        result.append({"text": ''.join(test_data[step]["words"]),
                       "label": decoder_entity})

    with open("./output/predict.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    predict()