import stimpy

if __name__ == '__main__':
    filename = 'slide-seq-normalized.n5'
    
    pucks = get_dataset_collection(filename)

    # focus on a puck
    print('\n')
    puck_name = get_dataset_names(pucks)[4]
    print('Focusing on puck,', puck_name)
    puck = get_dataset(pucks, puck_name)

    # get locations of the dataset
    locations = np.array(puck['locations'])
    print(locations.shape)
    
    # get expression vector for a gene
    gene = 'Hpca'
    print('\n')
    print('This is the gene expression vector for', gene)
    print(get_gene_expression(pucks, puck_name, gene))
    
    # plot gene expression
    plot_gene(pucks, puck_name, gene)

    plt.figure(figsize=(8, 16))
    count = 1

    # plot all aligned images
    for puck_name in get_dataset_names(pucks):
        puck = get_dataset(pucks, puck_name)
        locations = np.array(puck['locations'])
        expr = get_gene_expression(pucks, puck_name, gene)
        aligned_locations = get_aligned_locations(pucks, puck_name)

        plt.subplot(7,4,count)
        plt.scatter(locations[0, :], locations[1, :], c=expr, s=0.15)
        count += 1
        
        plt.subplot(7,4,count)
        plt.scatter(aligned_locations[0, :],
                    aligned_locations[1, :], c=expr, s=0.15)
        count += 1

    plt.show()
