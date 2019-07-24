# coding=utf-8
# @Author:yuanxiao and Google AI Language Team Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import sys
from bert import modeling
from bert import optimization
from bert import tokenization
from bert import tf_metrics
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_token, token_label=None, predicate_value_list=None, predicate_location_list=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_token = text_token
        self.token_label = token_label
        self.predicate_value_list = predicate_value_list
        self.predicate_location_list = predicate_location_list


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 token_label_ids,
                 predicate_matrix_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_label_ids = token_label_ids
        self.predicate_matrix_ids = predicate_matrix_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SKE_2019_Subject_Relation_Object_extraction_Processor(DataProcessor):
    """Processor for the SKE_2019 data set"""

    # SKE_2019 data from http://lic2019.ccf.org.cn/kg

    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), "r", encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "labeling_out.txt"), "r", encoding='utf-8') as labeling_out_f:
                with open(os.path.join(data_dir, "predicate_value_out.txt"), "r",
                          encoding='utf-8') as predicate_value_out_f:
                    with open(os.path.join(data_dir, "predicate_location_out.txt"), "r",
                              encoding='utf-8') as predicate_location_out_f:
                        token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                        token_label_out_list = [seq.replace("\n", '') for seq in labeling_out_f.readlines()]
                        predicate_value_out_list = [eval(seq.replace("\n", '')) for seq in
                                                    predicate_value_out_f.readlines()]
                        predicate_location_out_list = [eval(seq.replace("\n", '')) for seq in
                                                       predicate_location_out_f.readlines()]
                        examples = list(zip(token_in_list, token_label_out_list, predicate_value_out_list,
                                            predicate_location_out_list))
                        return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")

    def _raw_token_labels(self):
        return ['Date', 'Number', 'Text', '书籍', '人物', '企业', '作品', '出版社', '历史人物', '国家', '图书作品', '地点', '城市', '学校', '学科专业',
                '影视作品', '景点', '机构', '歌曲', '气候', '生物', '电视综艺', '目', '网站', '网络小说', '行政区', '语言', '音乐专辑']

    def get_token_labels(self):
        raw_token_labels = self._raw_token_labels()
        BIO_token_labels = ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]"]  # id 0 --> [Paddding]
        for label in raw_token_labels:
            BIO_token_labels.append("B-" + label)
            BIO_token_labels.append("I-" + label)
        BIO_token_labels.append("O")
        return BIO_token_labels

    def get_predicate_labels(self):
        "N --> no predicate"
        return ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
                '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
                '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
                '连载网站', '邮政编码', '面积', '首都']

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_token = line
                token_label = None
                predicate_value_list = None
                predicate_location_list = None
            else:
                text_token = line[0]
                token_label = line[1]
                predicate_value_list = line[2]
                predicate_location_list = line[3]
            examples.append(
                InputExample(guid=guid, text_token=text_token, token_label=token_label,
                             predicate_value_list=predicate_value_list, predicate_location_list=predicate_location_list))
        return examples


def convert_single_example(ex_index, example, token_label_list, predicate_label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            token_label_ids=[0] * max_seq_length,
            predicate_matrix_ids=[0] * (max_seq_length * max_seq_length),
            is_real_example=False)

    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    predicate_label_map = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_map[label] = i

    text_token = example.text_token.split(" ")
    if example.token_label is not None:
        token_label = example.token_label.split(" ")
    else:
        token_label = ["O"] * len(text_token)
    assert len(text_token) == len(token_label)


    if len(text_token) > (max_seq_length-2): #one for [CLS] and one for [SEP]
        text_token = text_token[0:max_seq_length-2]
        token_label = token_label[0:max_seq_length-2]

    if example.predicate_value_list is not None:
        predicate_value_list = example.predicate_value_list
        predicate_location_list =example.predicate_location_list
    else:
        predicate_value_list = [["N"] for _ in range(len(token_label))]
        predicate_location_list = [[i] for i in range(len(token_label))]

    predicate_matrix_ids = _get_multiple_predicate_matrix(predicate_label_map, predicate_value_list,
                                                          predicate_location_list, max_seq_length)

    tokens = []
    token_label_ids = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[CLS]"])

    for token, label in zip(text_token, token_label):
        tokens.append(token)
        segment_ids.append(0)
        token_label_ids.append(token_label_map[label])

    tokens.append("[SEP]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)


    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        token_label_ids.append(0)
        tokens.append("[Padding]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("token_label_ids: %s" % " ".join([str(x) for x in token_label_ids]))
        tf.logging.info("predicate_value_list: %s" % " ".join([str(x) for x in predicate_value_list]))
        tf.logging.info("predicate_location_list: %s" % " ".join([str(x) for x in predicate_location_list]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        token_label_ids=token_label_ids,
        predicate_matrix_ids=predicate_matrix_ids,
        is_real_example=True)
    return feature


def _get_multiple_predicate_matrix(predicate_label_map, predicate_value_list, predicate_location_list, max_seq_length):
    predicate_matrix = np.zeros((max_seq_length, max_seq_length), dtype=np.int32)
    for xi, predicate_value in enumerate(predicate_value_list):
        location_i = xi
        if "N" in predicate_value:
            continue
        for xj, value in enumerate(predicate_value):
            location_j = predicate_location_list[xi][xj]
            value_id = predicate_label_map[value]
            if location_i < max_seq_length and location_j < max_seq_length:
                predicate_matrix[location_i, location_j] = value_id
    return predicate_matrix


def file_based_convert_examples_to_features(
        examples, token_label_list, predicate_label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, token_label_list, predicate_label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_int_matrix_feature(matrix_ids):
            list_ids = matrix_ids.flatten()
            list_ids = list_ids.tolist()
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list_ids))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["token_label_ids"] = create_int_feature(feature.token_label_ids)
        features["predicate_matrix_ids"] = create_int_matrix_feature(feature.predicate_matrix_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "predicate_matrix_ids": tf.FixedLenFeature([seq_length * seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def getHeadSelectionScores(encode_input, hidden_size_n1, label_number):
    def broadcasting(left, right):
        left = tf.transpose(left, perm=[1, 0, 2])
        left = tf.expand_dims(left, 3)
        right = tf.transpose(right, perm=[0, 2, 1])
        right = tf.expand_dims(right, 0)
        B = left + right
        B = tf.transpose(B, perm=[1, 0, 3, 2])
        return B

    encode_input_hidden_size = encode_input.shape[-1].value
    u_a = tf.get_variable("u_a", [encode_input_hidden_size, hidden_size_n1])
    w_a = tf.get_variable("w_a", [encode_input_hidden_size, hidden_size_n1])
    v = tf.get_variable("v", [hidden_size_n1, label_number])
    b_s = tf.get_variable("b_s", [hidden_size_n1])

    left = tf.einsum('aij,jk->aik', encode_input, u_a)
    right = tf.einsum('aij,jk->aik', encode_input, w_a)
    outer_sum = broadcasting(left, right)
    outer_sum_bias = outer_sum + b_s
    output = tf.tanh(outer_sum_bias)
    g = tf.einsum('aijk,kp->aijp', output, v)
    return g


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 token_label_ids, predicate_matrix_ids, num_token_labels, num_predicate_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. float Tensor of shape [batch_size, hidden_size]
    # model_pooled_output = model.get_pooled_output()

    #     """Gets final hidden layer of encoder.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    #       to the final hidden of the transformer encoder.
    #     """
    sequence_bert_encode_output = model.get_sequence_output()
    if is_training:
        sequence_bert_encode_output = tf.nn.dropout(sequence_bert_encode_output, keep_prob=0.9)

    bert_sequenc_length = sequence_bert_encode_output.shape[-2].value
    sequnce_mask_length = tf.reduce_mean(tf.cast(input_mask, tf.float32)) * bert_sequenc_length
    sequnce_mask_length = tf.cast(sequnce_mask_length, tf.int32)

    with tf.variable_scope("predicate_head_select_loss"):
        # predicate_score_matrix = getHeadSelectionScores(encode_input=sequence_bert_encode_output, hidden_size_n1=100,
        #                                                 label_number=num_predicate_labels)
        def broadcasting(left, right):
            left = tf.transpose(left, perm=[1, 0, 2])
            left = tf.expand_dims(left, 3)
            right = tf.transpose(right, perm=[0, 2, 1])
            right = tf.expand_dims(right, 0)
            B = left + right
            B = tf.transpose(B, perm=[1, 0, 3, 2])
            return B

        encode_input = sequence_bert_encode_output
        hidden_size_n1 = 10
        label_number = num_predicate_labels
        encode_input_hidden_size = encode_input.shape[-1].value
        u_a = tf.get_variable("u_a", [encode_input_hidden_size, hidden_size_n1])
        w_a = tf.get_variable("w_a", [encode_input_hidden_size, hidden_size_n1])
        v = tf.get_variable("v", [hidden_size_n1, label_number])
        b_s = tf.get_variable("b_s", [hidden_size_n1])

        left = tf.einsum('aij,jk->aik', encode_input, u_a)
        right = tf.einsum('aij,jk->aik', encode_input, w_a)
        outer_sum = broadcasting(left, right)
        outer_sum_bias = outer_sum + b_s
        output = tf.tanh(outer_sum_bias)
        g = tf.einsum('aijk,kp->aijp', output, v)

        predicate_score_matrix = g
        head_select_scores_matrix_N_predicate = predicate_score_matrix[:, 0:sequnce_mask_length, 0:sequnce_mask_length, 0:1]
        head_select_scores_matrix_N_predicate_sum = tf.reduce_sum(head_select_scores_matrix_N_predicate)


        predicate_head_probabilities = tf.nn.sigmoid(predicate_score_matrix)
        # predicate_head_predictions_round = tf.round(predicate_head_probabilities)
        # predicate_head_predictions = tf.cast(predicate_head_predictions_round, tf.int32)
        predicate_head_predictions = tf.argmax(predicate_head_probabilities, axis=-1)
        predicate_matrix = tf.reshape(predicate_matrix_ids, [-1, bert_sequenc_length, bert_sequenc_length])
        gold_predicate_matrix_one_hot = tf.one_hot(predicate_matrix, depth=num_predicate_labels, dtype=tf.float32)

        predicate_score_matrix_masked = predicate_score_matrix[:, 0:sequnce_mask_length, 0:sequnce_mask_length, 1:]
        gold_predicate_matrix_one_hot_masked = gold_predicate_matrix_one_hot[:, 0:sequnce_mask_length, 0:sequnce_mask_length, 1:]
        predicate_sigmoid_cross_entropy_with_logits_masked = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=predicate_score_matrix_masked,
            labels=gold_predicate_matrix_one_hot_masked)
        predicate_head_select_loss = tf.reduce_sum(predicate_sigmoid_cross_entropy_with_logits_masked)
        # return predicate_head_probabilities, predicate_head_predictions, predicate_head_select_loss

    with tf.variable_scope("token_label_loss"):
        bert_encode_hidden_size = sequence_bert_encode_output.shape[-1].value
        token_label_output_weight = tf.get_variable(
            "token_label_output_weights", [num_token_labels, bert_encode_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        token_label_output_bias = tf.get_variable(
            "token_label_output_bias", [num_token_labels], initializer=tf.zeros_initializer()
        )
        sequence_bert_encode_output = tf.reshape(sequence_bert_encode_output, [-1, bert_encode_hidden_size])
        token_label_logits = tf.matmul(sequence_bert_encode_output, token_label_output_weight, transpose_b=True)
        token_label_logits = tf.nn.bias_add(token_label_logits, token_label_output_bias)

        token_label_logits = tf.reshape(token_label_logits, [-1, FLAGS.max_seq_length, num_token_labels])
        token_label_log_probs = tf.nn.log_softmax(token_label_logits, axis=-1)
        token_label_one_hot_labels = tf.one_hot(token_label_ids, depth=num_token_labels, dtype=tf.float32)
        token_label_per_example_loss = -tf.reduce_sum(token_label_one_hot_labels * token_label_log_probs, axis=-1)
        token_label_loss = tf.reduce_sum(token_label_per_example_loss)
        token_label_probabilities = tf.nn.softmax(token_label_logits, axis=-1)
        token_label_predictions = tf.argmax(token_label_probabilities, axis=-1)
        # return (token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predict)

    loss = predicate_head_select_loss + token_label_loss + head_select_scores_matrix_N_predicate_sum
    return (loss,
            predicate_head_select_loss, predicate_head_probabilities, predicate_head_predictions,
            token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions)


def model_fn_builder(bert_config, num_token_labels, num_predicate_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        token_label_ids = features["token_label_ids"]
        predicate_matrix_ids = features["predicate_matrix_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(token_label_ids), dtype=tf.float32)  # TO DO

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,
         predicate_head_select_loss, predicate_head_probabilities, predicate_head_predictions,
         token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            token_label_ids, predicate_matrix_ids, num_token_labels, num_predicate_labels,
            use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(predicate_head_select_loss, token_label_per_example_loss, token_label_ids, token_label_logits,
                          is_real_example):
                token_label_predictions = tf.argmax(token_label_logits, axis=-1, output_type=tf.int32)
                token_label_pos_indices_list = list(range(num_token_labels))[
                                               4:]  # ["[Padding]","[##WordPiece]", "[CLS]", "[SEP]"] + seq_out_set
                pos_indices_list = token_label_pos_indices_list[:-1]  # do not care "O"
                token_label_precision_macro = tf_metrics.precision(token_label_ids, token_label_predictions,
                                                                   num_token_labels,
                                                                   pos_indices_list, average="macro")
                token_label_recall_macro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="macro")
                token_label_f_macro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels,
                                                    pos_indices_list,
                                                    average="macro")
                token_label_precision_micro = tf_metrics.precision(token_label_ids, token_label_predictions,
                                                                   num_token_labels,
                                                                   pos_indices_list, average="micro")
                token_label_recall_micro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="micro")
                token_label_f_micro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels,
                                                    pos_indices_list,
                                                    average="micro")
                token_label_loss = tf.metrics.mean(values=token_label_per_example_loss, weights=is_real_example)
                predicate_head_select_loss = tf.metrics.mean(values=predicate_head_select_loss)
                return {
                    "predicate_head_select_loss": predicate_head_select_loss,
                    "eval_token_label_precision(macro)": token_label_precision_macro,
                    "eval_token_label_recall(macro)": token_label_recall_macro,
                    "eval_token_label_f(macro)": token_label_f_macro,
                    "eval_token_label_precision(micro)": token_label_precision_micro,
                    "eval_token_label_recall(micro)": token_label_recall_micro,
                    "eval_token_label_f(micro)": token_label_f_micro,
                    "eval_token_label_loss": token_label_loss,
                }

            eval_metrics = (metric_fn,
                            [predicate_head_select_loss, token_label_per_example_loss,
                             token_label_ids, token_label_logits, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "predicate_head_probabilities": predicate_head_probabilities,
                    "predicate_head_predictions": predicate_head_predictions,
                    "token_label_predictions": token_label_predictions},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ske_2019": SKE_2019_Subject_Relation_Object_extraction_Processor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    token_label_list = processor.get_token_labels()
    predicate_label_list = processor.get_predicate_labels()

    num_token_labels = len(token_label_list)
    num_predicate_labels = len(predicate_label_list)

    token_label_id2label = {}
    for (i, label) in enumerate(token_label_list):
        token_label_id2label[i] = label
    predicate_label_id2label = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_id2label[i] = label

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_token_labels=num_token_labels,
        num_predicate_labels=num_predicate_labels,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, token_label_list, predicate_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, token_label_list, predicate_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, token_label_list, predicate_label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        token_label_output_predict_file = os.path.join(FLAGS.output_dir, "token_label_predictions.txt")
        predicate_output_predict_file = os.path.join(FLAGS.output_dir, "predicate_head_predictions.txt")
        predicate_output_predict_id_file = os.path.join(FLAGS.output_dir, "predicate_head_predictions_id.txt")
        predicate_head_probabilities_file = os.path.join(FLAGS.output_dir, "predicate_head_probabilities.txt")
        with open(token_label_output_predict_file, "w", encoding='utf-8') as token_label_writer:
            with open(predicate_output_predict_file, "w", encoding='utf-8') as predicate_head_predictions_writer:
                with open(predicate_output_predict_id_file, "w", encoding='utf-8') as predicate_head_predictions_id_writer:
                    with open(predicate_head_probabilities_file, "w", encoding='utf-8') as predicate_head_probabilities_writer:
                        num_written_lines = 0
                        tf.logging.info("***** token_label predict and predicate labeling results *****")
                        for (i, prediction) in enumerate(result):
                            token_label_prediction = prediction["token_label_predictions"]
                            predicate_head_predictions = prediction["predicate_head_predictions"]
                            predicate_head_probabilities = prediction["predicate_head_probabilities"]
                            if i >= num_actual_predict_examples:
                                break
                            token_label_output_line = " ".join(
                                token_label_id2label[id] for id in token_label_prediction) + "\n"
                            token_label_writer.write(token_label_output_line)

                            predicate_head_predictions_flatten = predicate_head_predictions.flatten()
                            predicate_head_predictions_line = " ".join(
                                predicate_label_id2label[id] for id in predicate_head_predictions_flatten) + "\n"
                            predicate_head_predictions_writer.write(predicate_head_predictions_line)
                            #
                            predicate_head_predictions_id_line = " ".join(str(id) for id in predicate_head_predictions_flatten) + "\n"
                            predicate_head_predictions_id_writer.write(predicate_head_predictions_id_line)

                            # predicate_head_probabilities_flatten = predicate_head_probabilities.flatten()
                            # predicate_head_probabilities_line = " ".join(str(prob) for prob in predicate_head_probabilities_flatten) + "\n"
                            # predicate_head_probabilities_writer.write(predicate_head_probabilities_line)

                            num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
