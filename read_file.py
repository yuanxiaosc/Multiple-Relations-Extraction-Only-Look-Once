import os

data_dir = "bin/standard_format_data/train"

def get_examples(data_dir):
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
                    print(token_in_list[0])
                    print(token_label_out_list[0])
                    print(predicate_value_out_list[0])
                    print(predicate_location_out_list[0])
                    examples = list(
                        zip(token_in_list, token_label_out_list, predicate_value_out_list, predicate_location_out_list))
                    return examples

examples = get_examples(data_dir)
