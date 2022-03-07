# -*- coding: utf-8 -*-
# @Time: 2021/12/27 15:48
# @Author: zxf


import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default="./data/cluener/", type=str,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path",
                        # default="D:/pretrain_model/torch/chinese-bert-wwm-ext/",
                        default="D:/Spyder/pretrain_model/transformers_torch_tf/chinese-roberta-wwm-ext/",
                        # default="/opt/nlp/pretrain_model/chinese-roberta-wwm-ext/",
                        type=str,   # required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " )
    parser.add_argument("--output_dir", default="./output/",
                        type=str,   # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Other parameters
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--num_labels", default=1,
                        type=int,
                        help="label number")
    parser.add_argument("--soft_label", default=False,
                        type=bool,
                        help="soft label")
    parser.add_argument("--do_lower_case",   # action="store_true",
                        default=True, type=bool,
                        help="Set this flag if you are using an uncased model.")
    # model param
    parser.add_argument('--batch_size', default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. "
                             "Override num_train_epochs.", )
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_model_name", type=str,
                        default="span_ner_model.pt",
                        help="save model name")
    parser.add_argument("--label2id_file", default="label2id.json", type=str,
                        help="label encoder file")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gpu_ids", type=str, default="-1",
                        help="nivdia device number", )

    args = parser.parse_args()
    return args