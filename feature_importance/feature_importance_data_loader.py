#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# xinzhizheng @ 2021-02-29 19:07:52

import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row

Record = Row(
    "log_time",
    "guid",
    "doc_id",
    "query_id",
    "label",
    "biz_type",
    "feature",
    "ext_info")

SAMPLE_RATE = 0.0015
REPARTITION_NUM = 2000


class RankingSampleError(Exception):
    """Exception raised for errors about sample.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Ranking sample preprocess error"):
        self.message = message
        super().__init__(self.message)


def filter_field(is_using, as_fm_feature, slot_id, field_size):
    """whether field is filted
    :param is_using: field is using now
    :param as_fm_feature: field in fm part
    :param slot_id: numerous field flag
    :param field_size: field size
    :return: whether field is filted
    """
    return is_using == 1 and as_fm_feature == 1 and slot_id >= 0 and field_size > 0


def load_using_feature(using_feature_file='./using_feature.offline'):
    """load feature content from using_feature_file
    :param using_feature_file: feature content
    :return: inner_field_sizes: feature size dict
             inner_field_slot: feature slot dict
    """
    inner_field_sizes = dict()
    inner_field_slot = dict()
    with open(using_feature_file, 'r') as f:
        for line in f:
            contents = line.replace("\n", "").split("\t")
            if len(contents) < 10:
                print("FORMAT ERROR")
                continue
            feature_set_id = int(contents[1])
            is_using = int(contents[2])
            as_fm_feature = int(contents[4])
            slot_id = int(contents[7])
            field_size = int(contents[8])
            if filter_field(is_using, as_fm_feature, slot_id, field_size):
                inner_field_sizes[feature_set_id] = field_size
                inner_field_slot[feature_set_id] = slot_id

    return inner_field_sizes, inner_field_slot


def filter_sample(train_sample_sp):
    """filter useful or error sample
    :param train_sample_sp: origin sample
    :return: whether sample is filtered
    """
    # bad sample filter
    if len(train_sample_sp) < 35:
        return True

    # subchannal sample filter
    if len(train_sample_sp) >= 43 and len(train_sample_sp[42]) > 0\
            and train_sample_sp[42] not in ["news_news_top", "news_news_recommand"]:
        return True

    # pseudo sample filter
    pseudo_sample_opt = 2
    if len(train_sample_sp) >= 38 and len(train_sample_sp[37]) > 0\
            and (int(train_sample_sp[37]) & pseudo_sample_opt) > 0:
        return True

    return False


def get_hash_feature(feature_hash):
    """feature hashed convert to numerous format
    :param feature_hash: feature hashed
    :return: feature_concat: numerous format of hashed feature
    """
    feat_ids = feature_hash.split(";")
    feature_concat = []
    for fid in feat_ids:
        if len(fid) <= 0:
            continue

        feat_ids_vec = fid.split(",")
        if int(feat_ids_vec[0]) in field_sizes and field_sizes[int(feat_ids_vec[0])] != 0:
            key = (int(feat_ids_vec[1]) % (field_sizes[int(feat_ids_vec[0])] - 1)) + 1
            slot_id = field_slot[int(feat_ids_vec[0])] + 101
        else:
            continue

        if int(feat_ids_vec[0]) == 152:
            key = 0

        value = 1.0
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])
    return feature_concat


def get_dense_feature(train_sample_sp, feature_concat):
    """get dense feature(statistic feature)
    :param train_sample_sp: origin sample
    :param feature_concat: numerous feature list
    """
    user_stat = train_sample_sp[35].split(",")
    for i in range(len(user_stat)):
        key = i
        slot_id = 1024
        value = float(user_stat[i])
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])


def get_sequence_click_features(click_news, click_week, click_hour, click_cat1, click_cat2, feature_concat):
    # sequence preprocess for click_sequence
    click_sequence = click_news.split(",")
    num_seq = 0
    for i in range(min(len(click_sequence), 21)):
        key = click_sequence[i]
        slot_id = 2000 + i
        value = 1.0
        if key == 0:
            continue
        else:
            num_seq = i + 1
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])

    # click_sequence_week
    click_sequence_week = click_week.split(",")
    if len(click_sequence_week) > 0:
        click_sequence_week[0] = "0"
    for i in range(min(num_seq, 21)):
        key = click_sequence_week[i]
        slot_id = 2100 + i
        value = 1.0
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])

    # click_sequence_hour
    click_sequence_hour = click_hour.split(",")
    if len(click_sequence_hour) > 0:
        click_sequence_hour[0] = "0"
    for i in range(min(num_seq, 21)):
        key = click_sequence_hour[i]
        slot_id = 2200 + i
        value = 1.0
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])

    # click_sequence_cat1
    click_sequence_cat1 = click_cat1.split(",")
    for i in range(min(num_seq, 21)):
        key = click_sequence_cat1[i]
        slot_id = 2300 + i
        value = 1.0
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])

    # click_sequence_cat2
    click_sequence_cat2 = click_cat2.split(",")
    for i in range(min(num_seq, 21)):
        key = click_sequence_cat2[i]
        slot_id = 2400 + i
        value = 1.0
        feature_concat.append([str(key), int(slot_id), format(value, '.6f')])


def get_interact_label(train_sample_sp):
    if len(train_sample_sp[28]) > 0:
        actions = train_sample_sp[28].split(",")
    else:
        actions = []

    like = 1.0 if "1" in actions or "36" in actions else 0.0
    favorite = 1.0 if "2" in actions else 0.0
    comment = 1.0 if "4" in actions else 0.0
    share = 1.0 if "5" in actions else 0.0
    focus = float(train_sample_sp[33]) if len(train_sample_sp[33]) > 0 else 0.0
    return like, favorite, comment, share, focus


def is_video_sample(train_sample_sp):
    """whether sample is video
    :param train_sample_sp: origin sample
    :return: whether sample is video
    """
    article_type = int(train_sample_sp[32]) if len(train_sample_sp[32]) > 0 else -1

    video_types = {4, 101, 118, 303}
    is_video = 0
    if train_sample_sp[1][8] == 'V' or article_type in video_types:
        is_video = 1
    return is_video


def process_sample_data(line, log_date):
    """sample preprocessing and concat wuliang format sample(line level)
    :param line: origin sample
    :param log_date: hive date
    :return: Record object
    """
    train_sample_sp = line.split(" ")

    if filter_sample(train_sample_sp):
        return None

    flow_id, news_id, dev_id, label_long_click, label_short_click, \
        feature_hash, click_news, click_week, click_hour, click_cat1, click_cat2, _, _, _, \
        real_click, _, is_raise_hot, is_train_pcr, watch_time, video_len, _, _, video_play = train_sample_sp[:23]

    is_train_pcr = float(is_train_pcr)
    watch_time = float(watch_time)
    video_play = float(video_play)
    max_watch_time = max(watch_time, video_play)
    video_len = float(video_len)

    # get main feature(hashed feature)
    feature_concat = get_hash_feature(feature_hash)

    # get user statistic feature
    get_dense_feature(train_sample_sp, feature_concat)

    # get click sequence features(click_id, click_week, click_hour, click_cat1, click_cat2
    get_sequence_click_features(click_news, click_week, click_hour, click_cat1, click_cat2, feature_concat)

    # get news type: whether sample is video
    is_video = is_video_sample(train_sample_sp)

    # get interactions label
    like, favorite, comment, share, focus = get_interact_label(train_sample_sp)

    # add extra labels
    feature_concat.append([str(1), int(1000), format(float(label_long_click), '.6f')])
    feature_concat.append([str(1), int(1001), format(float(label_short_click), '.6f')])
    feature_concat.append([str(1), int(1005), format(float(real_click), '.6f')])
    feature_concat.append([str(1), int(1006), format(like, '.6f')])
    feature_concat.append([str(1), int(1007), format(favorite, '.6f')])
    feature_concat.append([str(1), int(1008), format(comment, '.6f')])
    feature_concat.append([str(1), int(1009), format(share, '.6f')])
    feature_concat.append([str(1), int(1011), format(focus, '.6f')])
    feature_concat.append([str(1), int(1012), format(float(is_video), '.6f')])
    feature_concat.append([str(1), int(1014), format(float(is_raise_hot), '.6f')])
    feature_concat.append([str(1), int(1017), format(video_len, '.6f')])
    feature_concat.append([str(1), int(1015), format(is_train_pcr, '.6f')])
    feature_concat.append([str(1), int(1016), format(watch_time, '.6f')])
    feature_concat.append([str(1), int(1020), format(video_play, '.6f')])
    feature_concat.append([str(1), int(1021), format(max_watch_time, '.6f')])

    feature_concat = sorted(feature_concat, key=lambda x: x[1])

    feature_list = []
    for key, slot_id, value in feature_concat:
        feature_list.append(str(key) + ":" + str(slot_id) + ":" + str(value))

    ext_list = ["real_click:{}".format(real_click)]

    # build Record object
    log_time = log_date
    guid = dev_id
    doc_id = news_id
    query_id = flow_id
    label = label_long_click
    biz_type = 2
    feature = ";".join(feature_list)
    ext_info = ";".join(ext_list)

    rec = Record(
        log_time,
        guid,
        doc_id,
        query_id,
        label,
        biz_type,
        feature,
        ext_info)
    return rec


def save_sample_to_hive(task_name, date, input_sample_path, table_name):
    """generate wuliang format input data and save in hive table
    :param task_name: spark task name
    :param date: table name date flag
    :param input_sample_path: input file path
    :param table_name: table name prefix
    :return: None
    """
    spark = SparkSession.builder.appName(task_name).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    print(input_sample_path + "/**/")
    temp_view = "sample_importance_score_format_view_{}_{}".format(str(date), table_name.split('.')[-1])
    print(temp_view)

    drop_sql = "alter table %s drop partition (ds=%s)" % (table_name, date)
    add_sql = "alter table %s add partition (ds=%s)" % (table_name, date)

    try:
        print(drop_sql)
        spark.sql(drop_sql)
    except Exception as e:
        print(e)
        raise RankingSampleError("Ranking sample preprocess error: drop sql failed")

    print(add_sql)
    spark.sql(add_sql)

    for i in range(24):
        try:
            if i < 10:
                hour = "0{}".format(str(i))
            else:
                hour = str(i)
            input_sample_full_path = "{}/{}".format(input_sample_path, hour)
            print(input_sample_full_path)
            sample_data = sc.textFile(input_sample_full_path, REPARTITION_NUM) \
                .sample(False, SAMPLE_RATE).map(lambda line: process_sample_data(line, date))\
                .filter(lambda x: x is not None)
            sample_data_df = sample_data.toDF()
            sample_data_df.printSchema()
            sample_data_df.repartition(REPARTITION_NUM).createOrReplaceTempView(temp_view)
            insert_sql = "insert into table %s partition (ds=%s) select * from %s" \
                         % (table_name, date, temp_view)
            print(insert_sql)
            spark.sql(insert_sql)
        except Exception as ex:
            print(ex)
            print("Error {}".format(input_sample_full_path))
            raise RankingSampleError("Ranking sample preprocess error: insert sql failed")

    print("save over")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("param err")
        sys.exit(1)

    date = sys.argv[1]
    input_sample_path = sys.argv[2]
    table_name = sys.argv[3]

    print("date: {}".format(date))
    print("input_sample_path: {}".format(input_sample_path))
    print("table_name: {}".format(table_name))

    date = int(date)
    TASK_NAME = "save_sample_to_hive_importance_score_format_table.xinzhizheng"

    try:
        field_sizes, field_slot = load_using_feature()
        save_sample_to_hive(TASK_NAME, date, input_sample_path, table_name)
    except RankingSampleError as e:
        print(e)
