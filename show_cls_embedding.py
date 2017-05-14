import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import imagenet
from nltk.corpus import wordnet as wn


SHOW_LABELS = False
VISU_SYNSETS = {
    'Geological formation': wn.synset('geological_formation.n.01'),
    'Natural object': wn.synset('natural_object.n.01'),
    'Sport': wn.synset('sport.n.01'),
    'Artifact': wn.synset('artifact.n.01'),
    'Fungus': wn.synset('fungus.n.01'),
    'Person': wn.synset('person.n.01'),
    'Animal': wn.synset('animal.n.01'),
}



syns = list(wn.all_synsets())
offsets_list = [(s.offset(), s) for s in syns]
offsets_dict = dict(offsets_list)
hyp = lambda s:s.hypernyms()

synsets = [offsets_dict[int(imagenet.CLASS_ID_TO_SYNSET[i][1:])] for i in xrange(1000)]

def get_category(s):
    h = s.hypernym_paths()[0]
    for category, other in VISU_SYNSETS.items():
        if other in h:
            return category
    return 'Other'

categories = np.array(map(get_category, synsets))





embedding = np.load('temp_ckpts/clsemb.npy')
model = TSNE(n_components=2, random_state=0)
points = model.fit_transform(embedding)


for category in set(categories):
    sel = points[categories == category]
    print category, len(sel)
    plt.plot(sel.T[0], sel.T[1], 'o', label=category)

plt.legend(prop={'size': 10})

if SHOW_LABELS:
    for i in xrange(1000):
        name = imagenet.CLASS_ID_TO_NAME[i][:33]
        plt.annotate(
            name,
            xy=points[i],
            xytext=(5, 0),
            textcoords='offset points')


plt.show()