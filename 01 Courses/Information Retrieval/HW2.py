#用於stem words
from nltk.stem import PorterStemmer

#用於計算log
import math

#用於讀檔、寫入檔案時path需要
import os.path


def Tokenize(DocID):
    # read txt file
    current_path = os.path.abspath(os.getcwd())
    save_path = current_path.replace('\\', '/') + '/documents/'
    f = open(save_path +str(DocID)+ ".txt", 'r')
    document = f.readlines()
    f.close()

    # split document with space
    words = []
    count = count2 = 0

    for line in document:
        count += 1

    for line in document:
        count2 += 1
        if count2 == count:
            line = line.lower()
            linewords = line.split(' ')
            words = words + linewords
        else:
            line = line.lower()
            linewords = line.split(' ')
            words = words + linewords

    # stem the splitted words and remove numbers
    words_stem = []
    for word in words:
        # remove puntuation
        word = word.strip('`-[]{};:"\,<>./?@#$%^&*_~')
        word = word.strip("'")

        # remove numbers
        contains_digit = any(map(str.isdigit, word))
        if contains_digit:
            continue
        word = PorterStemmer().stem(word)

        # remove puntuation again
        word = word.strip('`-[]{};:"\,<>./?@#$%^&*_~')
        words_stem.append(word.strip("'"))

    # define stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                 "as",
                 "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
                 "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
                 "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
                 "it's",
                 "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
                 "off",
                 "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                 "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
                 "that",
                 "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
                 "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
                 "up",
                 "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
                 "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
                 "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                 "yourself", "yourselves"]

    # remove stopwords
    result = []

    for word in words_stem:

        if word in stopwords:
            continue

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

        if word in stopwords:
            continue

        result.append(word)

    # return result
    return result


def tfidf(DocID, Dictionary):
    current_path = os.path.abspath(os.getcwd())
    save_path = current_path.replace('\\', '/') + '/tfidf'
    completeName = os.path.join(save_path, str(DocID) + ".txt")
    f1 = open(completeName, "w")

    Terms = Tokenize(DocID)
    dict_doc = {}
    for word in Terms:
        if word in dict_doc:
            dict_doc[word] += 1
        else:
            dict_doc[word] = 1

    dictlist = sorted(dict_doc.items(), key=lambda d: d[0])

    for item in dictlist:
        for i in range(len(Dictionary)):
            if Dictionary[i][1] == item[0]:
                f1.write("%4d  %s  %f\n" % (
                Dictionary[i][0], Dictionary[i][1], round(item[1] * math.log((1095 / Dictionary[i][2]), 10), 3)))
                break

    f1.close()


def UnitVector(DocID, Dictionary):
    current_path = os.path.abspath(os.getcwd())
    save_path = current_path.replace('\\', '/') + '/tfidf/'
    f = open(str(save_path) + str(DocID) + ".txt", "r")
    tfidf = f.readlines()

    length = 0

    for line in tfidf:
        items = line.split()
        length += float(items[-1]) ** 2

    vec = [0] * len(Dictionary)

    for line in tfidf:
        items = line.split()
        index = int(items[0])
        vec[index - 1] = float(items[-1]) / (length ** (1 / 2))
    return vec


def cosine(Docx, Docy):
    # 各別計算Docx, Docy的unit vector
    V1 = UnitVector(Docx, Dictionary)
    V2 = UnitVector(Docy, Dictionary)

    # 將兩unit vector內積得sim值
    sim = 0
    for i in range(len(Dictionary)):
        sim += V1[i] * V2[i]
    print("cosine for %d and %d is %f" %(Docx, Docy, sim))


# 主程式 for 第一小題

dictionary_df = {}

for i in range(1, 1096):

    # 存取各文件tokenize完的結果為dict_doc(list)，並忽略文件內重複的term。
    dict_doc = []
    for word in Tokenize(i):
        if word not in dict_doc:
            dict_doc.append(word)

    # 將各文件的dict_doc(list)內的字，出現即在dictionary_df+1，表有 +1 文件內有該term
    for word in dict_doc:
        if word in dictionary_df:
            dictionary_df[word] += 1
        else:
            dictionary_df[word] = 1

# 將dictionary_df結果依照terms的字母排序
dictionary_sort = sorted(dictionary_df.items(), key=lambda d: d[0])

# 將結果寫入 1.Dictionary(list) 2.dictionary.txt
Dictionary = []
i = 0
f = open("dictionary.txt", "w+")
for item in dictionary_sort:
    i += 1
    Dictionary.append([i, item[0], item[1]])
    f.write("%d  %s  %d\n" % (i, item[0], item[1]))
f.close()

# 主程式 for 第二小題

if not os.path.exists("tfidf"):
    os.mkdir("tfidf")

for i in range(1,1096):
    tfidf(i, Dictionary)

# 主程式 for 第三小題

cosine(1,2)