from top2vec import Top2Vec
import umap.plot
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def generate_documents_plot(self, background_color="white", reduced=False):
    """
    Create a documents and topics scatter plot.
    A documents and topics scatter plot will be generated and displayed.
    Documents on the same topic display in the same color. 
    On the plot, circles mean documents, and numbers mean the number of 
    topic. 
    Cautions: The plot should only be considered as approximate information
    because it is a result of dimention reduction from 5D to 2D. And it could
    be hard to distinguish for more than 20 topics. In this case, use the
    reduced option.
    Parameters
    ----------
    background_color : str (Optional, default='white')
        Background color for the plot. Suggested options are:
            * white
            * black
    reduced: bool (Optional, default False)
        Original topics are used by default. If True the
        reduced topics will be used.
    Returns
    -------
    A matplotlib plot of documents and topics
    """

    if reduced:
        self._validate_hierarchical_reduction()
        topic_vectors = self.topic_vectors_reduced
        topic_sizes, topic_nums = self.get_topic_sizes(reduced=True)
        doc_topics = self.doc_top_reduced
        doc_dist = self.doc_dist_reduced
    else:
        topic_vectors = self.topic_vectors
        topic_sizes, topic_nums = self.get_topic_sizes()
        doc_topics = self.doc_top
        doc_dist = self.doc_dist
    if len(topic_nums) <= 12:
        cmap = 'Paired'
    elif len(topic_nums) <= 20:
        cmap = 'plasma'
    else:
        cmap = 'hsv'
    # Args for UMAP. Same as current args except n_components to plot graph
    #umap_args_for_plot = self.umap_args.copy()
    umap_args_for_plot = umap_args
    # Dimension reduction
    document_vectors = self._get_document_vectors()
    topic_vectors = topic_vectors
    umap_model = umap.UMAP(**umap_args_for_plot, random_state=42).fit(document_vectors)  #  + self.topic_words
    document_vectors_2d = umap_model.embedding_  # same as umap_model.transform(self._get_document_vectors())
    topic_vectors_2d = umap_model.transform(topic_vectors)
    ## Draw a graph
    fig = plt.figure(figsize=(16, 12))
    ax = fig.subplots()
    ax.axis("off")
    fig.set_facecolor(background_color)
    # Draw document vectors
    ax.scatter(*document_vectors_2d.T, 
                s=20, c=doc_topics, cmap=cmap, alpha=0.7,  # linewidth=1, 
                )
    # Draw topic vectors
    ax.scatter(*topic_vectors_2d.T, 
                s=200, linewidth=2, c=topic_nums, cmap=cmap, alpha=0.4,
                )
    for topic_num, topic_vector_2d in zip(topic_nums, topic_vectors_2d):
        ax.annotate(topic_num, 
                    topic_vector_2d,
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=10,
                    color='white') 
    plt.show()
    

top2vec_model = Top2Vec.load('first_dry_run_with_opinions_from_1970_on')

umap_args = {
    "n_neighbors": 15,
    "n_components": 2,
    "metric": "cosine",
    "min_dist": 0.10,
    'spread': 1
}

generate_documents_plot(top2vec_model)
