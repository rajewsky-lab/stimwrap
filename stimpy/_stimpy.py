import numpy as np
import pyn5


def get_container(filename):
    """Read the container as object.

    Args:
        filename (string): the path to the container.

    Returns:
        pyn5 File: the container in pyn5 format.
    """
    container = pyn5.File(filename, mode=pyn5.Mode.READ_ONLY)
    print('Container contains the following', 
        container.attrs['numDatasets'], 'datasets:')
    print(container.attrs['datasets'])
    return container


def get_dataset_names(container):
    """Get all dataset names that are inside a container.

    Args:
        container (pyn5): [description].

    Returns:
        list of strings: List of dataset names.
    """
    return container.attrs['datasets']


def get_dataset(container, dataset_name):
    """Get a specific dataset from the container.

    Args:
        container (pyn5): the container where the dataset is inside.
        dataset_name (str): the name of the dataset.

    Returns:
        pyn5 Group: the dataset in pyn5 format.
    """
    return container.get(dataset_name)


def get_attribute_from_dataset(container, dataset_name, attribute='geneList'):
    """Get an attribute of a specific dataset.

    Args:
        container (pyn5): the container where the dataset is inside
        dataset_name (str): the name of the dataset.
        attribute (str, optional): the attribute to be returned. Available 
            options: 'geneList' (default), 'barcodeList', 'metadataList'.

    Returns:
        numpy array: the attribute to be returned.
    """
    dataset = container.get(dataset_name)
    return np.array(dataset.attrs[attribute])


def get_item_from_dataset(container, dataset_name, item='locations'):
    """Extract an item from a specific dataset.

    Args:
        container (pyn5): the container where the dataset is inside.
        dataset_name (str): the name of the dataset.
        item (str, optional): The item to be returned. Available options:
            'locations' (default), 'meta-celltype'.

    Returns:
        numpy array: the item to be returned.
    """
    dataset = container.get(dataset_name)
    return np.array(dataset[item])


def get_gene_expression_from_dataset(container, dataset_name, gene='all'):
    """Get gene expression from a specific dataset. It returns either a vector
    for a specific gene or the whole gene expression matrix if no gene is provided.

    Args:
        container (pyn5): the container where the dataset is inside.
        dataset_name (str): the name of the dataset.
        gene (str, optional): the gene to get the expression for. If  gene == 'all'
            (default) the whole gene expression matrix is returned.

    Returns:
        numpy array: gene expression vector or matrix
    """
    dataset = container.get(dataset_name)
    if gene != 'all':
        gene_idx = int(np.where(np.array(dataset.attrs['geneList']) == gene)[0])
        gene_expression = dataset['expression'][:, gene_idx]
        return gene_expression
    else:
        return np.round(np.array(dataset['expression']), 4)


def get_transform_matrix(container, dataset_name,
                         transformation='model_icp'):
    """Get the transformation matrix that is used to transform the original 
    locations to the aligned locations.

    Args:
        container (pyn5): the container where the dataset is inside.
        dataset_name (str): the name of the dataset.
        transformation (str, optional): the transformation used. Can be 
            'model_icp' (default), 'model_sift'.

    Returns:
        numpy array: the transform matrix.
    """
    dataset = container.get(dataset_name)
    transform_matrix = np.array(dataset.attrs[transformation]).reshape(2, 3)
    transform_matrix = np.concatenate((transform_matrix,
        np.array([0, 0, 1]).reshape(1, 3)))
    return transform_matrix


def get_aligned_locations(container, dataset_name, transformation='model_icp'):
    """Get the aligned locations of a dataset after having aligned it to the
    rest datasets in the container.

    Args:
        container (pyn5): the container where the dataset is inside.
        dataset_name (str): the name of the dataset.

    Returns:
        numpy array: the aligned locations.
    """
    locations = get_item_from_dataset(container, dataset_name, item='locations')
    num_locations = locations.shape[1]
    locations = np.concatenate((locations, 
        np.ones(num_locations).reshape(1, num_locations)))
    transform_matrix = get_transform_matrix(container,
                                            dataset_name,
                                            transformation=transformation)
    aligned_locations = np.dot(transform_matrix, locations)
    return aligned_locations[:2, :]
