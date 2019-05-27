#!/usr/bin/env python
# -*- coding: utf-8  -*-
import sys
import jieba
import jieba.analyse
import pandas


SORUCE_FILE = "clean_data.csv"
# jieba.analyse.set_stop_words("stopwords_dict.txt")


def positive_words(positive_comment_df):
    words_list = []
    for row in positive_comment_df:
        words = jieba.cut(row, cut_all=True, HMM=False)
        for word in words:
            words_list.append(word)
    return words_list


def negtive_words(negative_comment_df):
    words_list = []
    for row in negative_comment_df:
        words = jieba.cut(row, cut_all=True, HMM=False)
        for word in words:
            words_list.append(word)
    return words_list


def build_word_dict(positive_comment_df, negative_comment_df):
    pos_list = positive_words(positive_comment_df)
    neg_list = negtive_words(negative_comment_df)

    with open("positive_dict.txt", "w") as target:
        for word in pos_list:
            target.write(word + "\n")

    with open("negtive_dict.txt", "w") as target:
        for word in neg_list:
            target.write(word + "\n")


def ml_train(source_df):
    comment_df = source_df['comment']
    sentiment_df = source_df['sentiment']

    jieba_word_cut = lambda x: " ".join(jieba.cut(x)) 
    comment_df['word_cut'] = source_df.comment.apply(jieba_word_cut)
    print(comment_df)
    return 


def main():
    soruce_file_df =  pandas.read_csv(SORUCE_FILE, header=None, names=['a', 'b', 'comment', 'frequence', 'sentiment'])
    positive_comment_df = soruce_file_df[soruce_file_df['sentiment'] == 1]['comment']
    negative_comment_df = soruce_file_df[soruce_file_df['sentiment'] == 0]['comment']

    # 构建字典
    # build_word_dict(positive_comment_df, negative_comment_df)
    # soruce_file_df.shape

    ml_train(soruce_file_df)


    # with open("positive_comment.txt", "a") as target:
    #     target.write(positive_comment_df.to_string(index=False))


    # with open("negtive_comment.txt", "a") as target:
    #     target.write(negative_comment_df.to_string(index=False))

    # print(neg_comment_df)
    # comment_df = soruce_file_df[2]
    # product_frequency = soruce_file_df[3]
    # sentiment_df = soruce_file_df[4]
    # print(comment_df, product_frequency, sentiment_df)
    # pos_comment(sentiment_df)

if __name__ == "__main__":
    main()
    # print(neg_comment_df)
    # comment_df = soruce_file_df[2]
    # product_frequency = soruce_file_df[3]
    # sentiment_df = soruce_file_df[4]
    # print(comment_df, product_frequency, sentiment_df)
    # pos_comment(sentiment_df)

if __name__ == "__main__":
    main()