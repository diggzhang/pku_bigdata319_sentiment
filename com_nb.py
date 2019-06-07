#!/usr/bin/env python
# -*- coding: utf-8  -*-
import jieba # 用于中文分词
import pandas

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB


SORUCE_FILE = "clean_data.csv"
source_file_df =  pandas.read_csv(SORUCE_FILE, header=None, names=['a', 'b', 'comment', 'frequence', 'sentiment'])

X = source_file_df[['comment']]
y = source_file_df.sentiment

X['cutted_comment'] = X.comment.apply(lambda x: " ".join(jieba.cut(x)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=6000)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

count_vec = CountVectorizer()

x_count_train = count_vec.fit_transform(X_train.cutted_comment)
x_count_test = count_vec.transform(X_test.cutted_comment)

cnb_count = ComplementNB()
cnb_count.fit(x_count_train, y_train)
cnb_count_y_predict = cnb_count.predict(x_count_test)

print(
    cnb_count.score(x_count_test, y_test)
)
