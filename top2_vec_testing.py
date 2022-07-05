import pandas as pd
from top2vec import Top2Vec
import gzip

file = '/Users/spicy.kev/Documents/github/supreme_court_opinion_topic_modeling/data/supreme-court-opinions-by-author.jsonl.gz'

with gzip.open(file, 'rt') as fh:
    df = pd.read_json(fh, lines=True)

docs = df.text.tolist()
print(docs[0])
 
model = Top2Vec(docs)
 
topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)
print(topic_nums)
 
topic_words, word_scores, topic_nums = model.get_topics(10)
 
for words, scores, num in zip(topic_words, word_scores, topic_n ums):
    print(num)
    print(f"words: {words}")
 
documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=0, num_docs=10)
 
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------------------")
    print(doc)
    print("-----------------------")
    print()

'''
# Video 2
 
import pandas as pd
import re
from top2vec import Top2Vec
 
df = pd.read_json()
 
docs = df.descriptions.to_list()
print(docs[0])
print(docs[100])
 
docs = [d.replace('See ', "") for d in docs]
docs = [re.sub(r'\[^()]*\)', "", d).replace(" .", ".") for d in docs]
print(docs[100])
 
print(Top2Vec.__doc__)
 
model = Top2Vec(docs, speed='fast-learn')
 
model_learn = Top2Vec(docs, speed='learn')
 
model_dlearn = Top2Vec(docs, speed='deep-learn')
 
model_dlearn_with_workers = Top2Vec(docs, speed='deep-learn', workers=14)
 
topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)
print(topic_nums)
 
topic_sizes, topic_nums = model_learn.get_topic_sizes()
print(topic_sizes)
print(topic_nums)
 
topic_sizes, topic_nums = model_dlearn.get_topic_sizes()
print(topic_sizes)
print(topic_nums)
 
documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=50, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------------------")
    print(doc)
    print("-----------------------")
    print()
 
documents, document_scores, document_ids = model_learn.search_documents_by_topic(topic_num=286, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------------------")
    print(doc)
    print("-----------------------")
    print()
'''