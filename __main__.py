import stimpy

if __name__ == '__main__':
    filename = 'slide-seq-normalized.n5'
    
    pucks = stimpy.get_container(filename)

    # focus on a puck
    print('\n')
    puck_name = stimpy.get_dataset_names(pucks)[4]
    print('Focusing on puck,', puck_name)
    puck = stimpy.get_dataset(pucks, puck_name)

    # get locations directly from the dataset
    locations = np.array(puck['locations'])
    print(locations.shape)
    
    # alternatively fetch it from the container
    locations = stimpy.get_item_from_dataset(pucks, puck_name, item='locations')
    
    # get expression vector for a gene
    gene = 'Hpca'
    print('\n')
    print('This is the gene expression vector for', gene)
    print(stimpy.get_gene_expression_from_dataset(pucks, puck_name, gene=gene))
    
    # get the whole gene expression matrix of a dataset
    dge = stimpy.get_gene_expression_from_dataset(pucks, puck_name, gene='all')
    
    # get aligned locations of a puck
    aligned_locations = stimpy.get_aligned_locations(pucks, puck_name, 
                                                     transformation='model_sift')

