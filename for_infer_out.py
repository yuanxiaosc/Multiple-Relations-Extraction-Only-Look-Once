import os
import numpy as np
input_path = "infer_out/epochs6_weighted/ckpt2000"

token_label_predictions_file = open(os.path.join(input_path, "token_label_predictions.txt"), 'r', encoding='utf-8')
predicate_head_predictions_file = open(os.path.join(input_path, "predicate_head_predictions_id.txt"), 'r', encoding='utf-8')
# predicate_head_probabilities_file = open(os.path.join(input_path, "predicate_head_probabilities.txt"), 'r', encoding='utf-8')


def get_predicate_labels(self):
    "N --> no predicate"
    return ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
            '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
            '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
            '连载网站', '邮政编码', '面积', '首都']

def predicate_id2label_map():
    predicate_label_list = ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
            '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
            '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
            '连载网站', '邮政编码', '面积', '首都']
    predicate_label_id2label = dict()
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_id2label[i] = label
    return predicate_label_id2label



def get_subject_object_predicate_score_list_and_dict(i, j, predicate_head_id_matrix):
    label_list = ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
              '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
              '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
              '邮政编码', '面积', '首都']
    predicate_score_value_list = predicate_head_id_matrix[i, j]
    predicate_score_name_value_list = [(label, value) for label, value in zip(label_list, predicate_score_value_list)]
    predicate_score_name_value_sort_list = sorted(predicate_score_name_value_list, key=lambda x: x[1], reverse=True)
    predicate_score_name_value_dict = dict(predicate_score_name_value_sort_list)
    return predicate_score_name_value_sort_list, predicate_score_name_value_dict


def analysis_head_probabilities_file():
    M = 0
    for token_label, predicate_head in zip(token_label_predictions_file, predicate_head_probabilities_file):
        predicate_head_id_list = predicate_head.replace("\n", "").split(" ")
        predicate_head_id_list = [float(id) for id in predicate_head_id_list]
        predicate_head_id_matrix = np.array(predicate_head_id_list).reshape((128, 128, 50))
        predicate_score_name_value_sort_list, predicate_score_name_value_dict = get_subject_object_predicate_score_list_and_dict(2, 7, predicate_head_id_matrix)
        print(predicate_score_name_value_sort_list)
        print(predicate_score_name_value_dict)
        print("\n")
        M += 1
        if M > 3:
            break


def analysis_predicate_head_predictions_file(predicate_head_predictions_file):
    for token_label, predicate_head in zip(token_label_predictions_file, predicate_head_predictions_file):
        predicate_label_id2label = predicate_id2label_map()
        predicate_head_id_list = predicate_head.replace("\n", "").split(" ")
        predicate_head_id_list = [int(id) for id in predicate_head_id_list]
        predicate_head_id_matrix = np.array(predicate_head_id_list).reshape((128, 128, 50))
        predicate_location_i = 7
        predicate_location_j = 20
        # predicate_value_id = predicate_head_id_matrix[predicate_location_i, predicate_location_j]
        # predicate_value = predicate_label_id2label[predicate_value_id]
        # print(predicate_value)
        for i in range(128):
            for j in range(128):
                for k in range(1, 50):
                    if  predicate_head_id_matrix[i, j, k]!=0:
                        print(predicate_head_id_matrix[i,j])


analysis_predicate_head_predictions_file(predicate_head_predictions_file)