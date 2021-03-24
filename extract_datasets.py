from read_n5 import *

filename = 'slide-seq-normalized.n5'

pucks = get_dataset_collection(filename)

puck = get_dataset(pucks, get_dataset_names(pucks)[0])

locations = np.array(puck['locations'])
