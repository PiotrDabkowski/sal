# given a word returns a closest label
from nltk.corpus import wordnet as wn
from image_net_dataset import SYNSET_TO_CLASS_ID

AVAILABLE = None
def get_available():
    if AVAILABLE is not None: return
    print 'Getting all the synsets, may take a moment...'
    global AVAILABLE
    syns = list(wn.all_synsets())
    offsets_list = [(s.offset(), s) for s in syns]
    offsets_dict = dict(offsets_list)
    AVAILABLE = [offsets_dict[int(i[1:])] for i in SYNSET_TO_CLASS_ID]

def get_closest_label(word):
    targets = wn.synsets(word)
    if not targets:
        return None
    for t in targets:
        if word==t.name().split('.')[0]:
            target = t
            break
    else:
        return None
    get_available()
    return sorted(AVAILABLE, key=lambda x: target.path_similarity(x), reverse=True)[0]