import pandas as pd
import stimpy

filename = '/home/nkarais/Desktop/test_fiji/slide-seq-normalized.n5'
filename = '/home/nikos/Desktop/test_imglibst/out.n5'

pucks = stimpy.get_dataset_collection(filename)
puck_name = stimpy.get_dataset_names(pucks)[0]
puck = stimpy.get_dataset(pucks, puck_name)

print('Reading data from dataset', puck_name)

# expression = stimpy.get_dataset_expression(pucks, puck_name)
genes = stimpy.get_dataset_genes(pucks, puck_name)
locations = stimpy.get_dataset_locations(pucks, puck_name)
celltypes = stimpy.get_dataset_celltypes(pucks, puck_name)

print(celltypes)

# df = pd.DataFrame(data=expression.T, index=genes)
# df.to_csv('datasets/example')
# dfl = pd.DataFrame(data=locations.T)
