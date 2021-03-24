import numpy as np
import pyn5

def get_dataset_collection(filename):
    dataset_collection = pyn5.File(filename, mode=pyn5.Mode.READ_ONLY)
    print('Dataset collection contains the following', 
        dataset_collection.attrs['numDatasets'], 'datasets:')
    print(dataset_collection.attrs['datasets'])
    return dataset_collection

def get_dataset_names(dataset_collection):
    return dataset_collection.attrs['datasets']

def get_dataset(dataset_collection, dataset_name):
    return dataset_collection.get(dataset_name)

def get_dataset_expression(dataset_collection, dataset_name):
    dataset = dataset_collection.get(dataset_name)
    return np.array(dataset['expression'])

def get_dataset_locations(dataset_collection, dataset_name):
    dataset = dataset_collection.get(dataset_name)
    return np.array(dataset['locations'])

def get_gene_expression(dataset_collection, dataset_name, gene):
    dataset = dataset_collection.get(dataset_name)
    gene_idx = int(np.where(np.array(dataset.attrs['geneList']) == gene)[0])
    gene_expression = dataset['expression'][:, gene_idx]
    return gene_expression

def get_transform_matrix(dataset_collection, dataset_name,
                         transformation='model_icp'):
    dataset = dataset_collection.get(dataset_name)
    transform_matrix = np.array(dataset.attrs[transformation]).reshape(2, 3)
    transform_matrix = np.concatenate((transform_matrix,
        np.array([0, 0, 1]).reshape(1, 3)))
    return transform_matrix

def get_aligned_locations(dataset_collection, dataset_name):
    locations = get_dataset_locations(dataset_collection, dataset_name)
    num_locations = locations.shape[1]
    locations = np.concatenate((locations, 
        np.ones(num_locations).reshape(1, num_locations)))
    transform_matrix = get_transform_matrix(dataset_collection,
                                            dataset_name)
    aligned_locations = np.dot(transform_matrix, locations)
    return aligned_locations[:2, :]
