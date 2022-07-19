import numpy as np

#用於stem words
from nltk.stem import PorterStemmer

#用於計算log
import math

#用於讀檔、寫入檔案時需要
import os.path
import csv


def Tokenize(DocID):
    # read txt file
    current_path = os.path.abspath(os.getcwd())
    save_path = current_path.replace('\\', '/') + '/documents/'
    f = open(save_path + str(DocID) + ".txt", 'r')
    document = f.readlines()
    f.close()

    # split document with space
    words = []

    for line in document:
        line = line.lower()
        linewords = line.split(' ')
        words = words + linewords

    # stem the splitted words and remove numbers
    words_stem = []
    for word in words:

        # remvoe terms which are urls
        if word[:4] == "http":
            continue

        # remove puntuation
        word = word.strip('`-[]{};:"\,<>./?@#$%^&*_~0123456789')
        word = word.strip("'")

        word = PorterStemmer().stem(word)

        # remove puntuation again
        word = word.strip('`-[]{};:"\,<>./?@#$%^&*_~')
        words_stem.append(word.strip("'"))

    result = []

    for word in words_stem:

        # remove all puntuations in the word(not just at the end of the word)
        if word.isalpha() == False:
            s = ""
            for c in word:
                if c.isalpha():
                    s += c
            word = s
        # remove words which are too short
        if len(word) <= 1:
            continue

        result.append(word)

    # return result
    return result

def loglikelihood(dictionary_c):
    dictionary_l = []

    for word in dictionary_c:
        like = 0

        for j in range(1,14):

            N = 15*13
            n11 = word[j]
            n10 = 15 - n11
            n01 = sum(word[1:14]) - n11
            n00 = N - n01 - n11 - n10

            H1 = (((n11+n01)/N)**n11) * ((1-(n11+n01)/N)**n10) * (((n11+n01)/N)**n01) * ((1-(n11+n01)/N)**n00)
            H2 = ((n11/(n11+n10))**n11) * ((1-n11/(n11+n10))**n10) * ((n01/(n01+n00))**n01) * ((1-n01/(n01+n00))**n00)
            l = -2*(math.log(H1/H2))

            like += l
        dictionary_l.append(like)

    return dictionary_l


def MI(dictionary_c):
    dictionary_MI = []

    for word in dictionary_c:

        for j in range(1, 14):

            N = 15 * 13
            n11 = word[j]
            n10 = 15 - n11
            n01 = sum(word[1:14]) - n11
            n00 = N - n01 - n11 - n10

            p00 = n00 / N
            p01 = n01 / N
            p10 = n10 / N
            p11 = n11 / N

            if p00 == 0:
                P00 = 0
            else:
                P00 = p00 * (math.log(p00 / ((p01 + p00) * (p10 + p00))))

            if p01 == 0:
                P01 = 0
            else:
                P01 = p01 * (math.log(p01 / ((p01 + p11) * (p01 + p00))))

            if p10 == 0:
                P10 = 0
            else:
                P10 = p10 * (math.log(p10 / ((p10 + p11) * (p10 + p00))))

            if p11 == 0:
                P11 = 0
            else:
                P11 = p11 * (math.log(p11 / ((p11 + p01) * (p10 + p11))))

            MI = P00 + P01 + P10 + P11

        dictionary_MI.append(MI)
    return dictionary_MI


def MIX(likelihood, MI, dictionary_c):
    dictionary_MIX = []

    for i in range(len(dictionary_c)):
        dictionary_MIX.append([likelihood[i] + MI[i], dictionary_c[i][0]])

    return dictionary_MIX


def MutualNB(cdata):
    words = []
    features = np.zeros(500)

    for i in cdata:
        words = words + Tokenize(i)
    for word in words:
        if word in feature:
            features[feature.index(word)] += 1
    SUM = sum(features)
    V = np.count_nonzero(features)

    PTerm = np.zeros(500)
    for i in range(500):
        PTerm[i] = (features[i] + 1) / (SUM + V)

    return PTerm


def argmaxC(docID):
    # 因為P(C)都是1/13，所以這邊只計算P(x=t|C)
    P = np.zeros(13)

    for word in Tokenize(docID):
        if word in feature:
            index = feature.index(word)
            P[0] += math.log(PT1[index])
            P[1] += math.log(PT2[index])
            P[2] += math.log(PT3[index])
            P[3] += math.log(PT4[index])
            P[4] += math.log(PT5[index])
            P[5] += math.log(PT6[index])
            P[6] += math.log(PT7[index])
            P[7] += math.log(PT8[index])
            P[8] += math.log(PT9[index])
            P[9] += math.log(PT10[index])
            P[10] += math.log(PT11[index])
            P[11] += math.log(PT12[index])
            P[12] += math.log(PT13[index])
    m = max(P)
    c = list(P).index(m) + 1
    return c


# 讀檔，分出train, test 的 DocID

f = open("training.txt", 'r')
document = f.readlines()
f.close()

train = []
for line in document:
    docs = line.split(' ')
    for ele in docs[1:-1]:
        train.append(int(ele))

test = []
for i in range(1, 1096):
    if i not in train:
        test.append(i)

# 單存term的dictionary(dictionary內的term不重複)
dictionary = []

# 用於存各term出現在各class次數的dictionary (dictionary內的term不重複)
dictionary_c = []

# 用於存各term出現在哪個class的dictionary (dictionary內的term重複，讀到一次就加一筆)
dictionary_doc = []

# 讀取train Docs內的term並記錄各term出現在各class的次數
j = 0
for i in train:
    words = list(set(Tokenize(i)))
    for word in words:

        # 對於未出現過的字，加入dictionary中，
        if word not in dictionary:
            dictionary.append(word)
            dictionary_c.append([word, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 不論之前有無出現過，有新紀錄就加入dictionary_doc中
        dictionary_doc.append([word, int(j / 15) + 1])
    j += 1

# 將各term出現過的紀錄(dictionary_doc)，統一存入dictionary_c中，即統計各term在各class訓練集中出現的次數
for word in dictionary_doc:
    dictionary_c[dictionary.index(word[0])][word[1]] += 1

#計算dictionary_c中terms的loglikelihood, EMI值，並以1:500加權令為MIX值
like = loglikelihood(dictionary_c)
mi = MI(dictionary_c)
mix = MIX(like, mi, dictionary_c)

#將MIX值由大至小排序，並取前500作為features
d = sorted(mix, reverse = True)
feature = []
for word in d[:500]:
    feature.append(word[1])
feature = sorted(feature)

#用訓練集的文章計算各class中各feature的P(t)
PT1 = MutualNB(train[:15])
PT2 = MutualNB(train[15:30])
PT3 = MutualNB(train[30:45])
PT4 = MutualNB(train[45:60])
PT5 = MutualNB(train[60:75])
PT6 = MutualNB(train[75:90])
PT7 = MutualNB(train[90:105])
PT8 = MutualNB(train[105:120])
PT9 = MutualNB(train[120:135])
PT10 = MutualNB(train[135:150])
PT11 = MutualNB(train[150:165])
PT12 = MutualNB(train[165:180])
PT13 = MutualNB(train[180:])

# 預測test DocID並將預測class存入csv檔中
with open('output9.csv', 'w', newline='') as csvfile:
    # 定義欄位
    fieldnames = ['Id', 'Value']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in test:
        writer.writerow({'Id': i, 'Value': argmaxC(i)})