# -*- coding: utf-8 -*-
'''
论文来源：Graph Clustering Based on Structural/Attribute Similarities（VLDB09）
'''
import json
import logging
import sys


# reload(sys)
# sys.setdefaultencoding( "utf-8" )
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

# 输入：txt文件路径；输出：graph
def loadGraph(input):
    G = nx.Graph()
    f = open(input, 'r')
    #next(f)
    j = 0
    for line in f:
        rows = line.split(' ')
        if len(rows) == 3:
            node1 = rows[0]
            node2 = rows[1]
            Score = float(rows[2])
            checkNum = node1.isdigit() or node2.isdigit()
            #过滤低相似度边
            if Score<0.1 or node2 == ' ' or node1 == ' 'or node2 == '\t' or node1 == '\t':
                continue
            G.add_edge(node1, node2, weight=Score)
            j += 1
            if j%100000==0:
                logging.info('%s edges loaded' % (j))
        # if j>200000:break
    f.close()
    return G

#剪纸，输出图gml文件
def prun(input):
    G = nx.Graph()
    f = open(input, 'r')
    j = 0
    for line in f:
        rows = line.split(' ')
        node1 = rows[0]
        node2 = rows[1]
        Score = float(rows[2])
        checkNode = node1.isdigit() or node2.isdigit() or len(node1) <2 or len(node2) < 2
        if not checkNode:
            if Score>0.15 and node2 != ' ' and node1 != ' 'and node2 != '\t' and node1 != '\t':
                G.add_edge(node1, node2, weight=Score)
                j += 1
                if j % 100000 == 0:
                    logging.info('%s edges loaded' % (j))
            # if j > 200000: break
    #nx.write_gml(G,"/home/fanfan/ML_RUI/model/word_graph.gml.gz")
    Gm = nx.adjacency_matrix(G)
    save_sparse_csr("graphMatrix",Gm)
    logging.info('graph loaded,totally %s edges' % (j))

#SAC算法入参：graph:G ; length limit:l ; random walk restart probability:c ; influence function parameter:sig
#训练得到聚类簇中心，按influence density排序
def SAC(G,l,c):
    # 转化为稀疏矩阵，记录顶点信息
    # logging.info('graph loaded')
    #G = nx.read_gml('/home/fanfan/ML_RUI/model/word_graph1.gml')
    Gm = nx.adjacency_matrix(G)
    logging.info('adjacency matrix loaded')
    save_sparse_csr("graphMatrix",Gm)
    nodeNum = {}
    i = 0
    for n in G.nodes_iter():
        nodeNum[i] = n
        i += 1
    with open("/home/fanfan/ML_RUI/model/nodesInfo.json","a") as f:
        json.dump(nodeNum,f,ensure_ascii=False)
        f.write('\n')
    # random distance matrix
    walkpro = c*(1-c)
    rdm =Gm.copy()
    rdmsum = walkpro*rdm
    #rdmsum = walkpro*rdm
    if l>1:
        for mi in xrange(2,l+1):
            rdm = rdm.dot(Gm)
            walkpro *= (1-c)
            rdmsum += walkpro*rdm
            logging.info('walk %s finished'%(mi))
    mean = csr_matrix.mean(rdm) #距离平均值
    #array = (rdmsum.data-mean)**2
    #var = array.sum()/rdmsum.data.size #距离方差
    rdmarray = rdmsum.toarray()
    IFdic = {}
    num = 0
    for row in rdmarray:
        sumf = 0.0
        for element in xrange(1,row.size):
            if num == element: continue
            fenl = 1- np.e**(pow(row[element],2)/2/mean/-mean)
            sumf += fenl
        wordname = nodeNum[num]
        #influence function dictionary
        IFdic[wordname] = sumf
        num += 1
    #IFdic = sorted(IFdic.items(), key=lambda d: d[1],reverse=True)
    with open('/home/fanfan/ML_RUI/model/clusters/IFdic_part03.json', 'a') as outfile:
        json.dump(IFdic, outfile, ensure_ascii=False)
        outfile.write('\n')
    # topk = []
    # k = 20
    # for ele in IFdic:
    #     if k>0:
    #         topk.append(nodeNum[ele[0]])
    #         k -= 1
    #     else: break
    # with open('/home/fanfan/ML_RUI/model/topkwords.json', 'a') as outfile:
    #     json.dump(topk, outfile, ensure_ascii=False)
    #     outfile.write('\n')
    #print ''

#图聚类，各种簇单独计算得分，最终得分最高简历为推荐简历
def simOfWords(words,topK):
    from gensim import models
    wordsimModel = models.Word2Vec.load("/home/fanfan/ML_RUI/modelsmall/model.mm")
    cvInfos = open("/home/fanfan/ML_RUI/modelsmall/cvInfo.json","r")
    cvInfos = json.load(cvInfos)
    cvScore = {}
    for key,value in cvInfos.items():
        #1、获得匹配词
        samewords = wordMatch(words,value)
        #2、分配簇,并根据簇中心权重计算相似度
        score = np.float64(0)
        if samewords:
            for sameword in samewords:
                sm = np.float64(0)
                for center,weight in topK.items():
                    smn = wordsimModel.similarity(center,sameword)
                    if smn> sm:
                        sm = smn
                        whichCenter = center
                    score += topK[whichCenter]
        cvScore[key] = score
    cvScore = sorted(cvScore.items(), key=lambda d: d[1], reverse=True)
    for dd in cvScore:
        bestCV = cvInfos[dd[0]]
        break
    print bestCV

#词组间匹配词
def wordMatch(words1,words2):
    samewords = []
    if len(words1)>1 and len(words2)>1:
        for word in words1:
            if word in words2:
                samewords.append(word)
    if len(words1)==1 and len(words2)>1:
        if words1[0] in words2:
            samewords.append(words1[0])
    if len(words1)>1 and len(words2)==1:
        if words2[0] in words1:
            samewords.append(words2[0])
    if len(words1) == 1 and len(words2) == 1:
        if words1[0] == words2[0]:
            samewords.append(words1[0])
    return samewords

#保存稀疏矩阵到本地
def save_sparse_csr(filename, array):
    np.savez("/home/fanfan/ML_RUI/model/matrix/%s.npz"%(filename), data=array.data, indices=array.indices,indptr=array.indptr, shape=array.shape)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,filename='SAC.log', filemode='w')
    prun('/home/fanfan/ML_RUI/model/simInfo.txt')

