|PyPI|

.. |PyPI| image:: https://img.shields.io/pypi/v/stim-wrapper.svg
   :target: https://pypi.org/project/stim-wrapper/

stimpy - a python interface for STIM
====================================

``stimpy`` is a Python wrapper for the spatial transcriptomics library 
`STIM <https://github.com/PreibischLab/imglib2-st>`_. It provides an interface 
for extracting datasets and their attributes from ``n5`` containers that are 
created by STIM.

``stimpy`` is created and maintained by `Nikos Karaiskos <mailto:nikolaos.karaiskos@mdc-berlin.de>`_.

Installation
------------
To install the ``stim-wrapper`` try::

    pip install stim-wrapper

or if you do not have sudo rights::

    pip install --user stim-wrapper

Check if the library is successfully installed::

    import stim-wrapper as sw

Usage
-----
The following assumes that the file ``container.n5`` contains the datasets and their
attributes as created by ``STIM``.::

    pucks = sw.get_container('/path/to/container.n5')

Print the names of the datasets::

    print(sw.get_dataset_names(pucks))

Focus on a specific puck and extract the relevant information::

    puck_name = sw.get_datset_names(pucks)[0]
    puck = sw.get_datset(pucks, puck_name)

Get the puck locations either directly from the puck::

    locations = puck['locations']

or fetch them from the container::

    locations = sw.get_item_from_datset(pucks, puck_name, item='locations')

Fetch gene expression
~~~~~~~~~~~~~~~~~~~~~
It is possible to get the expression vector of a single gene::

    hpca_vec = sw.get_gene_expression_from_dataset(pucks, puck_name, gene='Hpca')

or the whole gene expression matrix::

    dge = sw.get_gene_expression_from_dataset(pucks, puck_name, gene='all')

Fetch dataset attributes
~~~~~~~~~~~~~~~~~~~~~~~~
``STIM`` stores the dataset attributes in the ``n5`` container. These can 
be directly accessed with ``stimpy``::

    sw.get_attribute_from_dataset(container, puck_name, attribute='geneList')

Available options also include: `barcodeList` and `metadataList`.

Fetch aligned locations
~~~~~~~~~~~~~~~~~~~~~~~
In the case where multiple consecutive sections are obtained and aligned with
``STIM``, the aligned locations can be obtained with::

    aligned_locations = sw.get_aligned_locations(pucks, puck_name,
                                                     transformation='model_sift')