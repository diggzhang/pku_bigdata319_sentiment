#!/usr/bin/env python
# -*- coding: utf-8  -*-
import sys
import jieba
import pandas
import jieba.analyse


SORUCE_FILE = "clean_data.csv"


def positive_words(positive_comment_df):
    for row in positive_comment_df:
        words = jieba.cut(row)
        for word in words:
            print(word)

def negtive_words():
    return 

def main():
    soruce_file_df =  pandas.read_csv(SORUCE_FILE, header=None, names=['a', 'b', 'comment', 'frequence', 'sentiment'])
    positive_comment_df = soruce_file_df[soruce_file_df['sentiment'] == 1]['comment']
    negative_comment_df = soruce_file_df[soruce_file_df['sentiment'] == 0]['comment']

    with open("positive_comment.txt", "a") as target:
        target.write(positive_comment_df.to_string(index=False))

    positive_words(positive_comment_df)

    with open("negtive_comment.txt", "a") as target:
        target.write(negative_comment_df.to_string(index=False))

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