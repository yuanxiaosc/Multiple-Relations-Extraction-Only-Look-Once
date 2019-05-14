import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()

def getHeadSelectionScores(encode_input, hidden_size_n1, predicate_number):
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
    v = tf.get_variable("v", [hidden_size_n1, predicate_number])
    b_s = tf.get_variable("b_s", [hidden_size_n1])

    left = tf.einsum('aij,jk->aik', encode_input, u_a)
    right = tf.einsum('aij,jk->aik', encode_input, w_a)
    outer_sum = broadcasting(left, right)
    outer_sum_bias = outer_sum + b_s
    output = tf.tanh(outer_sum_bias)
    g = tf.einsum('aijk,kp->aijp', output, v)
    return g

batch_size = 2
sequnce_length = 3
init_dimension = 5
headselection_mid_level_dimension = 7
predicate_label_number = 4
input_mask = tf.constant([[1, 1, 0], [1, 1, 0]])

sequnce_mask_length = tf.reduce_mean(tf.cast(input_mask, tf.float32)) * sequnce_length
sequnce_mask_length = tf.cast(sequnce_mask_length, tf.int32)
print("sequnce_mask_length:\t", sequnce_mask_length)

encode_input = tf.constant(np.random.random(size=(batch_size, sequnce_length, init_dimension)), dtype=tf.float32)
print("encode_input:\t", encode_input)

head_select_scores_matrix = getHeadSelectionScores(encode_input, headselection_mid_level_dimension, predicate_label_number)
print("head_select_scores_matrix:\t", head_select_scores_matrix)

#predicate_head_predictions = tf.argmax(head_select_scores_matrix, axis=-1)
#print("predicate_head_predictions:\t", predicate_head_predictions)
head_select_scores_matrix_N_predicate = head_select_scores_matrix[:,0:sequnce_mask_length, 0:sequnce_mask_length, 0:1]
print("head_select_scores_matrix_N_predicate:\t", head_select_scores_matrix_N_predicate)
head_select_scores_matrix_N_predicate_sum = tf.reduce_sum(head_select_scores_matrix_N_predicate)
print("head_select_scores_matrix_N_predicate_sum:\t", head_select_scores_matrix_N_predicate_sum)

head_select_sigmoid_scores_matrix = tf.nn.sigmoid(head_select_scores_matrix)
print("head_select_sigmoid_scores_matrix:\t", head_select_sigmoid_scores_matrix)
predicate_head_predictions = tf.round(head_select_sigmoid_scores_matrix)
predicate_head_predictions = tf.cast(predicate_head_predictions, tf.int32)
print("predicate_head_predictions:\t", predicate_head_predictions)


gold_label = tf.constant(np.random.randint(0, predicate_label_number, size=(batch_size, sequnce_length, sequnce_length)), dtype=tf.int32)
print("gold_label:\t", gold_label)
gold_label_one_hot = tf.one_hot(gold_label, depth=predicate_label_number, dtype=tf.float32)
print("gold_label_one_hot:\t", gold_label_one_hot)

head_select_scores_matrix_masked = head_select_scores_matrix[:, 0:sequnce_mask_length, 0:sequnce_mask_length, :]
gold_label_one_hot_masked = gold_label_one_hot[:, 0:sequnce_mask_length, 0:sequnce_mask_length, :]
print("head_select_scores_matrix_masked:\t", head_select_scores_matrix_masked)
print("gold_label_one_hot_masked:\t", gold_label_one_hot_masked)

sigmoid_cross_entropy_with_logits_masked = tf.nn.sigmoid_cross_entropy_with_logits(logits=head_select_scores_matrix_masked, labels=gold_label_one_hot_masked)
print("sigmoid_cross_entropy_with_logits_masked:\t", sigmoid_cross_entropy_with_logits_masked)



head_select_loss_masked = tf.reduce_sum(sigmoid_cross_entropy_with_logits_masked)
print("head_select_loss_masked:\t", head_select_loss_masked)

head_select_loss_add_N = head_select_loss_masked + head_select_scores_matrix_N_predicate_sum