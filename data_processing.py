# -*- coding: utf-8 -*-
# @Time    : 2022/1/9 17:19
# @Author  : zxf
import os
import json


def get_test_data_demo(data_file, output_file):
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            data.append(line["text"])

    result = data[:20]
    with open(output_file, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    data_file = "D:/Spyder/data/nlpdata/命名实体识别数据集/cluener_public/train.json"
    output_file = "./data/cluener/test_demo.json"
    get_test_data_demo(data_file, output_file)