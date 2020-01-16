import codecs
from collections import OrderedDict, deque, Counter
import itertools
import csv
import random
import argparse
import copy
import pickle as pkl
import h5py
import numpy as np

def loadMapping():
    paperToId = {}
    repoToId = {}
    idToPaper = {}
    idToRepo = {}
    
    with codecs.open('data/paper_id_map.csv', 'r', encoding = 'utf-8') as paper_id_map_file:
        csv_reader = csv.DictReader(paper_id_map_file, delimiter=',')
        for row in csv_reader:
            paperToId[row['paper_original_id']] = int(row['paper_id'])
            idToPaper[int(row['paper_id'])] = row['paper_original_id']
            
    with codecs.open('data/repo_id_map.csv', 'r', encoding = 'utf-8') as repo_id_map_file:
        csv_reader = csv.DictReader(repo_id_map_file, delimiter=',')
        for row in csv_reader:
            repoToId[row['repo_original_id']] = int(row['repo_id'])
            idToRepo[int(row['repo_id'])] = row['repo_original_id']
    print('Mapping loaded')
    
    return paperToId, idToPaper, repoToId, idToRepo

def loadGroundTruth():
    positives = {}
    with codecs.open('data/paper_repo_pair_ground_truth.csv', 'r', encoding = 'utf-8') as ground_truth_file:
        csv_reader = csv.reader(ground_truth_file, delimiter=',')
        for row in csv_reader:
            row = [int(x) for x in row]
            if np.mean(row[2:]) >= 1:
                if row[0] not in positives:
                    positives[row[0]] = []
                positives[row[0]].append(row[1])
    print('\tpaper with positives:', len(positives))
    return positives

def loadData():
    paperInfo = []
    repoInfo = []

    print('Reading dataset')
                
    with codecs.open('data/paper_title_keywords_abstract.csv', 'r', encoding = 'utf-8') as paper_info_file:
        csv_reader = csv.DictReader(paper_info_file, delimiter=',')
        for row in csv_reader:
            paperInfo.append({ 
                'id': int(row['paper_id']),
                'title': row['title'].split(),
                'titleCleaned': row['title_cleaned'].split(),
                'keywords': row['keywords'].split('#'),
                'abstract': row['abstract'].split(), 
                'abstractCleaned': row['abstract_cleaned'].split()
            })
    
    with codecs.open('data/repo_tags_description_url.csv', 'r', encoding = 'utf-8') as repo_info_file:
        csv_reader = csv.DictReader(repo_info_file, delimiter=',')
        for row in csv_reader:
            repoInfo.append({
                'id': int(row['repo_id']),
                'tagWords': sorted(list(set(list(itertools.chain(*[x.split() for x in row['tags'].split('#') if x]))))),
                'tags': list(set([x.strip() for x in row['tags'].split('#') if x])),
                'description': row['description'].split(),
                'descriptionCleaned': row['description_cleaned'].split(),
                'url': row['url']
            })

    print('\t{0:d} papers and {1:d} repositories'.format(len(paperInfo), len(repoInfo)))
    
    positives = loadGroundTruth()
    
    abstractBag = [v['abstractCleaned'] for v in paperInfo]
    descriptionBag = [v['descriptionCleaned'] for v in repoInfo]
    tagsBag = [v['tagWords'] for v in repoInfo]
    vocabularyBag = list(itertools.chain(*(abstractBag))) + list(itertools.chain(*(descriptionBag))) + list(itertools.chain(*(tagsBag)))
    vocabulary = OrderedDict((w, i) for i, w in enumerate(sorted(list(set(vocabularyBag)))))
    print('\t{0:d} words in vocabulary'.format(len(vocabulary)))
    
    print('\tAverage {0:.2f} title words, {1:.2f} paper keywords, {2:.2f} abstract words, {3:.2f} tags, and {4:.2f} description words'.format(
        np.mean([len(v['titleCleaned']) for v in paperInfo]), 
        np.mean([len(v['keywords']) for v in paperInfo]), 
        np.mean([len(v['abstractCleaned']) for v in paperInfo]), 
        np.mean([len(v['tags']) for v in repoInfo]),
        np.mean([len(v['descriptionCleaned']) for v in repoInfo])))

    return paperInfo, repoInfo, vocabulary, positives


def loadGraphData(paperInfo, repoInfo):
    print('Reading graph data')
    
    paperGraphAdjList = [OrderedDict() for _ in range(len(paperInfo))]
    repoGraphAdjList = [OrderedDict() for _ in range(len(repoInfo))]
    
    print('\tMaking paper citation connections')
    with codecs.open('data/paper_citation_graph.csv', 'r', encoding = 'utf-8') as paper_graph_file:
        csv_reader = csv.reader(paper_graph_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            js = list(map(lambda x: int(x), row))
            for j in js[1:]:
                paperGraphAdjList[js[0]][j] = None
                paperGraphAdjList[j][js[0]] = None
    
    print('\tMaking star repository connections')
    with codecs.open('data/repo_freq_graph.csv', 'r', encoding='utf-8') as repo_graph_file:
        csv_reader = csv.reader(repo_graph_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            js = list(map(lambda x: int(x), row))
            for j in js[1::2]:
                repoGraphAdjList[js[0]][j] = None
                repoGraphAdjList[j][js[0]] = None
            
    print('\t\tpaper citation graph size {0:d} (avg deg: {1:.2f})'.format(len(paperGraphAdjList), np.mean([len(s) for s in paperGraphAdjList])))
    print('\t\trepos stars graph size {0:d} (avg deg: {1:.2f})'.format(len(repoGraphAdjList), np.mean([len(s) for s in repoGraphAdjList])))

    bridgesLen = 0
    with codecs.open('data/paper_repo_bridge.csv', 'r', encoding = 'utf-8') as bridge_file:
        csv_reader = csv.reader(bridge_file, delimiter=',')
        next(csv_reader)
        for _ in csv_reader:
            bridgesLen += 1

    print('\t{0:d} valid paper-repo pairs'.format(bridgesLen))

    return paperGraphAdjList, repoGraphAdjList, bridgesLen

def connectSimilarRepo(repoGraphAdjList, repoInfo, threshold=0.4):
    print('Repository-repository connections')
    repoVocabMap = {k: i for i, k in enumerate(sorted(list(set(list(itertools.chain(*[x['tagWords'] + x['description'] for x in repoInfo]))))))}
    print('\tRepo vocab size {0:d}'.format(len(repoVocabMap)))
    
    print('\tCreating TF matrix')
    tfMat = np.zeros((len(repoInfo), len(repoVocabMap)))
    for i, r in enumerate(repoInfo):
        doc = r['tagWords'] + r['description']
        for k, v in dict(Counter(doc)).items():
            tfMat[i][repoVocabMap[k]] = v / len(doc)
    
    print('\tCreating IDF vector')
    idfVec = np.zeros(len(repoVocabMap))
    for r in repoInfo:
        for w in set(r['tagWords'] + r['description']):
            idfVec[repoVocabMap[w]] += 1
    idfVec = np.log((np.ones(len(idfVec)) * len(repoInfo)) / (1 + idfVec))
    
    print('\tCreating TF-IDF matrix')
    tfidfMat = tfMat * idfVec
    print('\tCreating document similarity matrix')
    tfidfVecs = np.array([np.sqrt(np.sum(np.power(tfidfMat, 2), axis=1))])
    normMat = tfidfVecs.T @ tfidfVecs
    docSimMat = ((tfidfMat @ tfidfMat.T) / normMat) >= threshold
    np.fill_diagonal(docSimMat, 0)
    
    progress = 0
    resultAdjList = copy.deepcopy(repoGraphAdjList)
    for i in range(0, len(repoInfo)):
        progress += 1
        print('\tMaking connection ... {0:d}%'.format(int(progress * 100 / len(repoInfo))), end='\r')
        for j in range(i + 1, len(repoInfo)):
            if docSimMat[i, j]:
                resultAdjList[i][j] = None
                resultAdjList[j][i] = None
    print('')
    print('\t\trepos full graph size {0:d} (avg deg: {1:.2f})'.format(len(resultAdjList), np.mean([len(s) for s in resultAdjList])))
    return resultAdjList

def readGloveEmbedding():
    '''read glove embedding'''
    word2vec = dict()
    fileName = "data/glove.6B.200d.txt"
    with open(fileName, "r") as f:
        for num, line in enumerate(f):
            arr = line.split()
            word = arr[0]
            word2vec[word] = [float(i) for i in arr[1:]]

    print("\tLength of embedding", len(arr) - 1)
    return word2vec

def getSelectedEmbedding(vocab, word2vec):
    '''get selected embedding'''
    arr = np.zeros(200)
    for w in word2vec:
        arr += np.array(word2vec[w])
    avg_embed = arr / len(word2vec)
    ## get the embedding of vocabulary
    selectedEmbedding = []
    selectedEmbedding.append(np.zeros(200))
    count = 0
    for w in vocab:
        if w in word2vec:
            selectedEmbedding.append(np.array(word2vec[w]))
            count += 1
        else:
            selectedEmbedding.append(avg_embed + random.random() * 0.0001)

    print("\tSelected vocabulary:", count)
    result = np.array(selectedEmbedding)
    print("\tTotal of %d selected words" % len(result))
    return result

if __name__ == '__main__':
    # Process arguments
    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1')
    parser = argparse.ArgumentParser(description = 'P2R Data Preprocessing')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-kaw', nargs = '?', type = int, default = 200, help = 'Number of words in abstract')
    parser.add_argument('-kdw', nargs = '?', type = int, default = 50, help = 'Number of words in description')
    parser.add_argument('-rrt', nargs = '?', type = float, default = 0.3, help = 'Repository-repository connection threshold')
    args = parser.parse_args()

    paperToId, idToPaper, repoToId, idToRepo = loadMapping()
    paperInfo, repoInfo, vocabulary, positives = loadData()
    paperGraphAdjList, coforkRepoGraphAdjList, bridgesLen = loadGraphData(paperInfo, repoInfo)
    repoGraphAdjList = connectSimilarRepo(coforkRepoGraphAdjList, repoInfo, args.rrt)
    
    paperGraphAdjList = list(map(lambda x: list(x.keys()), paperGraphAdjList))
    coforkRepoGraphAdjList = list(map(lambda x: list(x.keys()), coforkRepoGraphAdjList))
    repoGraphAdjList = list(map(lambda x: list(x.keys()), repoGraphAdjList))
    
    Npaper = len(paperGraphAdjList)
    Nrepo = len(repoGraphAdjList)
    
    paperFeatures = -np.ones((Npaper, args.kaw), dtype=int)
    repoFeatures = -np.ones((Nrepo, args.kdw), dtype=int)

    maxWordCount = 0
    maxTags = 0
    for repo in repoInfo:
        if len(repo['tags']) > maxTags:
            maxTags = len(repo['tags'])
        for tagwords in repo['tags']:
            tmpSplit = tagwords.split()
            if len(tmpSplit) > maxWordCount:
                maxWordCount = len(tmpSplit)
    repoTags = -np.ones((len(repoInfo), maxTags, maxWordCount), dtype=int)

    print('Max tag length: {0:d}, max tag number: {1:d}'.format(maxWordCount, maxTags))
    
    for i in range(len(paperInfo)):
        for j, word in enumerate(paperInfo[i]['abstractCleaned'][:args.kaw]):
            paperFeatures[i][j] = vocabulary[word]
    for i in range(len(repoInfo)):
        for j, word in enumerate(repoInfo[i]['descriptionCleaned'][:args.kdw]):
            repoFeatures[i][j] = vocabulary[word]
        for j, tagwords in enumerate(repoInfo[i]['tags']):
            for k, word in enumerate(tagwords.split()):
                repoTags[i, j, k] = vocabulary[word]
    
    print('Extracting embedding')
    word2vec = readGloveEmbedding()
    selectedEmbedding = getSelectedEmbedding(list(vocabulary.keys()), word2vec)
    
    print('Writing data')
    with open('data/ind.paper-repo.data', 'wb') as handle:
        pkl.dump({
            'paperGraphAdjList': paperGraphAdjList,
            'coforkRepoGraphAdjList': coforkRepoGraphAdjList,
            'repoGraphAdjList': repoGraphAdjList,
            'bridgeLength': bridgesLen,
            'bridgeIds': sorted(set(range(bridgesLen)) - set(positives.keys())),
            'paperFeatures': paperFeatures + 1,
            'repoFeatures': repoFeatures + 1,
            'repoTags': repoTags + 1,
            'positives': positives,
            'wordEmbeddings': selectedEmbedding
        }, handle)
