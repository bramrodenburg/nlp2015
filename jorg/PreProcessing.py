import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import string
import unicodedata

PRINTABLE = set(('Lu', 'Ll', 'Nd', 'Zs'))


def filter_non_printable(s):
    result = []
    for c in s:
        c = unicodedata.category(c) in PRINTABLE and c or u'#'
        result.append(c)
    return u''.join(result).replace(u'#', u' ')


def processFileFirst(inFilestr):

    inFile = open(inFilestr, 'r')
    outFile = open("F:/temp/edited.xml", 'w')
    for line in inFile:
        editline = filter_non_printable(line)
        outFile.write(editline)

    inFile.close()
    outFile.close()



def preProcessing(xdoc):

    stops = stopwords.words("english")
    reviews = extractAllSentences(xdoc, stops)
    print ("# of reviews %s" % (len(reviews)))
    # for review in reviews:
        # print "# of sentences in reviews %s" % (len(review))
        # for sentence in review:
            # print sentence


def extractAllSentences(xdoc, stops):

    reviewSentences = []
    reviews = []

    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(xdoc, parser=parser)
    # tree = ET.parse(xdoc)
    root = tree.getroot()
    # assuming file has <data> ... </data> as root tag, so must be added
    numOfReviews = 1
    for child in root:
        for review in child.findall("review_text"):
            # for each review text: 1) split in sentences
            #                               2) get rid of punctuations
            #                               3) tokenize to words
            #                               4) remove stop words
            reviewSentences = nltk.sent_tokenize(review.text)
            reviewSentences = [nltk.word_tokenize(sent.translate(None, string.punctuation)) for sent in reviewSentences]
            reviewSentencesWStops = []
            for sentence in reviewSentences:
                woStops = [word for word in sentence if word not in stops]
                reviewSentencesWStops.append(woStops)
            reviews.append(reviewSentencesWStops)
            numOfReviews += 1

    return reviews


if __name__ == '__main__':

    #inFile = "C:\\Users\\psftadm\\Study\\030 - Artificial Intelligence\\N - NLP\\010 - NLP I\\P - Project\\D - data\\sorted_data\\dvd\\all.review.xml"
    inFile = "F:/temp/dvdReviews.xml"
    # preProcessing(inFile)
    processFileFirst(inFile)
