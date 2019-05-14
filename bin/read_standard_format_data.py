import os
import numpy as np
import tensorflow as tf
from prettytable import PrettyTable
import collections
input_data_path = "standard_format_data/train"

#检查文件行数是否匹配
def check_file_line_number(input_data_path="standard_format_data/train"):
    text_f = open(os.path.join(input_data_path, "text.txt"), "r", encoding='utf-8')
    token_in_f = open(os.path.join(input_data_path, "token_in.txt"), "r", encoding='utf-8')
    token_in_not_UNK_f = open(os.path.join(input_data_path, "token_in_not_UNK.txt"), "r", encoding='utf-8')
    labeling_out_f = open(os.path.join(input_data_path, "labeling_out.txt"), "r", encoding='utf-8')
    predicate_value_out_f = open(os.path.join(input_data_path, "predicate_value_out.txt"), "r", encoding='utf-8')
    predicate_location_out_f = open(os.path.join(input_data_path, "predicate_location_out.txt"), "r", encoding='utf-8')
    format_data_file_list = [text_f, token_in_f, token_in_not_UNK_f, labeling_out_f, predicate_value_out_f,
                             predicate_location_out_f]
    file_lines_number = len(format_data_file_list[0].readlines())
    for file in format_data_file_list[1:]:
        file_lines_number_new = len(file.readlines())
        if file_lines_number_new != file_lines_number:
            raise ValueError("Error, file line number mismatch!")
        file.close()
#check_file_line_number(format_data_file_list)


#显示文件内容
def show_format_data(text, token_in, token_in_not_UNK, labeling_out, predicate_value_out, predicate_location_out):
    t = PrettyTable(["Key", "Value"])
    t.add_row(["text", text])
    t.add_row(["token_in", token_in])
    t.add_row(["token_in_not_UNK", token_in_not_UNK])
    t.add_row(["labeling_out", labeling_out])
    t.add_row(["predicate_value_out", predicate_value_out])
    t.add_row(["predicate_location_out", predicate_location_out])
    print(t)

#输出竖排格式数据
def output_vertical_format(token_in_not_UNK, labeling_out, predicate_value_out, predicate_location_out, show=True):
    token_in_not_UNK_list = token_in_not_UNK.split(" ")
    labeling_out_list = labeling_out.split(" ")
    predicate_value_out_list = eval(predicate_value_out)
    predicate_location_out_list = eval(predicate_location_out)
    assert len(token_in_not_UNK_list) == len(labeling_out_list)
    assert len(predicate_value_out_list) == len(predicate_location_out_list)
    assert len(token_in_not_UNK_list) == len(predicate_value_out_list)
    if show:
        t = PrettyTable(["index", "token", "label", "predicate value", "predicate location"])
        for idx, toke, labeling, predicate_value, predicate_location in \
                zip(range(len(token_in_not_UNK_list)), token_in_not_UNK_list, labeling_out_list,
                    predicate_value_out_list, predicate_location_out_list):

            t.add_row([idx, toke, labeling, predicate_value, predicate_location])
        print(t)

#生成关系映射字典
def get_predicate_label_map():
    #"N --> no predicate"
    predicate_label_list =  ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
            '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
            '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
            '连载网站', '邮政编码', '面积', '首都']
    predicate_label_map = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_map[label] = i
    return predicate_label_map

#获取关系矩阵
def get_multiple_predicate_matrix(predicate_label_map, predicate_value_list, predicate_location_list, max_seq_length, show=False):
    predicate_matrix = np.zeros((max_seq_length, max_seq_length), dtype=np.int32)
    for xi, predicate_value in enumerate(predicate_value_list):
        location_i = xi
        if "N" in predicate_value:
            continue
        for xj, value in enumerate(predicate_value):
            location_j = predicate_location_list[xi][xj]
            value_id = predicate_label_map[value]
            if show and location_j < max_seq_length and location_i < max_seq_length:
                predicate_matrix[location_i, location_j] = value_id
                #print(predicate_matrix)
                print("({},{})--({},{})".format(location_i, location_j, value, value_id))
                print("\n")
    return predicate_matrix


def predicate_matrix_2_tf_train_example(matrix_ids, output_file="predicate_matrix_tf_file"):
    list_ids = matrix_ids.flatten()
    list_ids = list_ids.tolist()

    writer = tf.python_io.TFRecordWriter(output_file)
    def create_int_matrix_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    features = collections.OrderedDict()
    features["predicate_matrix_ids"] = create_int_matrix_feature(list_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    writer.close()

def iter_format_data(input_data_path="standard_format_data/train"):
    predicate_label_map = get_predicate_label_map()
    max_seq_length = 128
    text_f = open(os.path.join(input_data_path, "text.txt"), "r", encoding='utf-8')
    token_in_f = open(os.path.join(input_data_path, "token_in.txt"), "r", encoding='utf-8')
    token_in_not_UNK_f = open(os.path.join(input_data_path, "token_in_not_UNK.txt"), "r", encoding='utf-8')
    labeling_out_f = open(os.path.join(input_data_path, "labeling_out.txt"), "r", encoding='utf-8').readlines()
    predicate_value_out_f = open(os.path.join(input_data_path, "predicate_value_out.txt"), "r", encoding='utf-8')
    predicate_location_out_f = open(os.path.join(input_data_path, "predicate_location_out.txt"), "r", encoding='utf-8')
    for text, token_in, token_in_not_UNK, labeling_out, predicate_value_out, predicate_location_out in\
            zip(text_f, token_in_f, token_in_not_UNK_f, labeling_out_f, predicate_value_out_f, predicate_location_out_f):
        #show_format_data(text, token_in, token_in_not_UNK, labeling_out, predicate_value_out, predicate_location_out)
        output_vertical_format(token_in_not_UNK, labeling_out, predicate_value_out, predicate_location_out, show=True)
        predicate_value_list = eval(predicate_value_out.replace("\n", ""))
        predicate_location_list = eval(predicate_location_out.replace("\n", ""))
        predicate_matrix = get_multiple_predicate_matrix(predicate_label_map, predicate_value_list, predicate_location_list, max_seq_length, show=True)
        predicate_matrix_2_tf_train_example(predicate_matrix)
iter_format_data()