from top2vec import Top2Vec
import umap.plot
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


top2vec_model = Top2Vec.load('first_dry_run_with_opinions_from_1970_on')

vectors = top2vec_model._get_document_vectors()
reduced2d = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', verbose=True).fit(vectors)

reduced_vectors = top2vec_model.topic_vectors_reduced
#labels = top2vec_model.doc_top_reduced

combined_vectors = np.vstack((top2vec_model.topic_vectors, top2vec_model._get_word_vectors()))
# If the Top2Vec version on git:
#combined_vectors = np.vstack((top2vec_model.topic_vectors, top2vec_model.word_vectors))

#top2vec_model.topic_words

with open('topic_words.txt', 'w') as f:
    f.write(str(top2vec_model.topic_words))

'''
umap_args_for_plot = {
    "n_neighbors": 10,
    "n_components": 2,
    "metric": "cosine",
    "min_dist": 0.10,
    'spread': 1
}

# Assuming we want to label by type, could also label by closest topic number
type_labels = []
for index in range(len(combined_vectors)):
    if index < len(top2vec_model.topic_vectors):
        type_labels.append('topic')
    else:
        type_labels.append('term')
umap_plot_mapper = umap.UMAP(**umap_args_for_plot).fit(combined_vectors)
umap.plot.points(umap_plot_mapper, labels=np.array(type_labels), theme='fire')
plt.show()


topic_nums = top2vec_model.get_num_topics()
#print(topic_nums)

topic_sizes, topic_nums = top2vec_model.get_topic_sizes()
print(f"The size of the topics found is: {topic_sizes}")
print("=========================")
print(f"The topic numbers found are: {topic_nums}")
print("=========================") 
#print(top2vec_model.topic_words)

umap_args_model = {
    "n_neighbors": 10,
    "n_components": 2,
    "metric": "cosine",
    "min_dist": 0.10,
    'spread': 1
}

documents, document_scores, document_ids = top2vec_model.search_documents_by_topic(topic_num=51, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    #print("-----------")
    #print(doc)
    print("-----------")


#umap_model = umap.UMAP(**umap_args_model).fit(top2vec_model._get_document_vectors(norm=False))
#umap_figure = umap.plot.points(umap_model, labels=top2vec_model.doc_top_reduced, theme='fire')
#plt.show()

#umap.plot.points(reduced2d, labels=labels, color_key_cmap = 'viridis', background='white')


#umap.plot.points(reduced2d, labels=topic_labels, color_key_cmap='viridis', background='white')
#plt.show()

#print(topic_nums)
#print(top2vec_model.get_num_topics())
#print(top2vec_model.topic_words)



topic_sizes, topic_nums = top2vec_model.get_topic_sizes()
print(f"The size of the topics found is: {topic_sizes}")
print("=========================")
print(f"The topic numbers found are: {topic_nums}")
print("=========================") 


topic_words, word_scores, topic_nums = top2vec_model.get_topics(50)
 
for words, scores, num in zip(topic_words, word_scores, topic_nums):
    print(f"The numbers are: {num}")
    print("=========================")
    print(f"The words are: {words}")


topic_words, word_scores, topic_scores, topic_nums = top2vec_model.search_topics(keywords=["abortion"], num_topics=25)
print(topic_nums)
for topic_words, word_scores, topic_scores, topic_nums in zip(topic_words, word_scores, topic_scores, topic_nums):
    print(topic_words, word_scores, topic_scores, topic_nums)
    #print("-----------")
    #print(doc)
    #print("-----------")
'''
