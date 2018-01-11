#!/usr/bin/python
# coding=utf-8
# 采用TextRank方法提取文本关键词
import sys
import pandas as pd
import jieba.analyse
"""
       TextRank权重：

            1、将待抽取关键词的文本进行分词、去停用词、筛选词性
            2、以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
            3、计算图中节点的PageRank，注意是无向带权图
"""

# 处理标题和摘要，提取关键词
def getKeywords_textrank(data,topK):
    idList,titleList,abstractList = data['id'],data['title'],data['abstract']
    ids, titles, keys = [], [], []
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index], abstractList[index]) # 拼接标题和摘要
        jieba.analyse.set_stop_words("data/stopWord.txt") # 加载自定义停用词表
        print "\"",titleList[index],"\"" , " 10 Keywords - TextRank :"
        keywords = jieba.analyse.textrank(text, topK=topK, allowPOS=('n','nz','v','vd','vn','l','a','d'))  # TextRank关键词提取，词性筛选
        word_split = " ".join(keywords)
        print word_split
        keys.append(word_split.encode("utf-8"))
        ids.append(idList[index])
        titles.append(titleList[index])

    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])
    return result

def main():
    dataFile = 'data/sample_data.csv'
    data = pd.read_csv(dataFile)
    result = getKeywords_textrank(data,10)
    result.to_csv("result/keys_TextRank.csv",index=False)

if __name__ == '__main__':
    main()
