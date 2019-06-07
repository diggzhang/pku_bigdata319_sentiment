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
    # %% md
    # 中文情感预测
    # %%
    # !/usr/bin/env python
    # -*- coding: utf-8  -*-
    import sys
    import jieba  # 用于中文分词
    import pandas

    # 默认提供的数据集data.csv在macOS下直接预读乱码，转存utf-8一份
    SORUCE_FILE = "clean_data.csv"
    # %%
    # 从源数据中主要提取comment列和sentiment列
    source_file_df = pandas.read_csv(SORUCE_FILE, header=None, names=['a', 'b', 'comment', 'frequence', 'sentiment'])
    # %%
    source_file_df.head()
    # %%
    # 以comment列内容为属性
    X = source_file_df[['comment']]
    # 以sentiment列内容为lable，分类只有两类0消极或1积极
    y = source_file_df.sentiment
    X.shape, y.shape
    # print(X, y)
    # %%
    # 调包侠关键步骤：使用jieba抽取comment列的内容进行分词，分词结果放到cutted_comment列中
    X['cutted_comment'] = X.comment.apply(lambda x: " ".join(jieba.cut(x)))
    # 可以看出comment列分词后放到了cutted_comment
    X.head()
    # %%
    # 使用train_test_split将数据集切分，按照作业要求取6000条数据用于训练
    # 肉眼看train_test_split默认给打乱数据集了？？？？
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=6000)
    # %%
    # 6000个训练属性，对应6000个标签
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    # %%
    # 文本向量化处理，sklearntt提供两个方法：CountVectorizer TfidfVectorizer
    # 选择CountVectorizer，原因：1. 数据集似乎是精心准备，不用做复杂的停用词处理 2. 搜到的多数教程以CountVectorizer为例，选此少踩坑
    # 主要参考博客 https://www.cnblogs.com/Lin-Yi/p/8974108.html
    from sklearn.feature_extraction.text import CountVectorizer
    count_vec = CountVectorizer()

    # %%
    x_count_train = count_vec.fit_transform(X_train.cutted_comment)
    x_count_test = count_vec.transform(X_test.cutted_comment)
    # %%
    # 使用朴素贝叶斯分类器  分别对两种提取出来的特征值进行学习和预测
    from sklearn.naive_bayes import MultinomialNB
    mnb_count = MultinomialNB()
    mnb_count.fit(x_count_train, y_train)  # 学习
    mnb_count_y_predict = mnb_count.predict(x_count_test)  # 预测
    # %%
    mnb_count.score(x_count_test, y_test)
    # %%


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
