|PyPI|

.. |PyPI| image:: https://img.shields.io/pypi/v/stimwrap.svg
   :target: https://pypi.org/project/stimwrap/

stimwrap - a python interface for STIM
======================================

``stimwrap`` is a Python wrapper for the spatial transcriptomics library 
`STIM <https://github.com/PreibischLab/STIM/>`_. It provides an interface 
for extracting datasets and their attributes from ``n5`` containers that are 
created by STIM.

``stimwrap`` is created and maintained by `Nikos Karaiskos <mailto:nikolaos.karaiskos@mdc-berlin.de>`_.

Installation
------------
To install the ``stimwrap`` try::

    pip install stimwrap

or if you do not have sudo rights::

    pip install --user stimwrap

Check if the library is successfully installed::

    import stimwrap as sw

If installation fails due to conflicting dependencies, create a dedicated environment
with ``python=3.7`` and try again to install ``stimwrap`` as above.

Usage
-----
The following assumes that the file ``container.n5`` contains the datasets and their
attributes as created by ``STIM``::

    pucks = sw.get_container('/path/to/container.n5')

Print the names of the datasets::

    print(sw.get_dataset_names(pucks))

Focus on a specific puck and extract the relevant information::

    puck_name = sw.get_datset_names(pucks)[0]
    puck = sw.get_datset(pucks, puck_name)

Get the puck locations either directly from the puck::

    locations = puck['locations']

or fetch them from the container::

    locations = sw.get_item_from_dataset(pucks, puck_name, item='locations')

Fetch gene expression
~~~~~~~~~~~~~~~~~~~~~
It is possible to get the expression vector of a single gene::

    hpca_vec = sw.get_gene_expression_from_dataset(pucks, puck_name, gene='Hpca')

or the whole gene expression matrix::

    dge = sw.get_gene_expression_from_dataset(pucks, puck_name, gene='all')

Fetch dataset attributes
~~~~~~~~~~~~~~~~~~~~~~~~
``STIM`` stores the dataset attributes in the ``n5`` container. These can 
be directly accessed with ``stimwrap``::

    sw.get_attribute_from_dataset(pucks, puck_name, attribute='geneList')

Available options also include: `barcodeList` and `metadataList`.

Fetch aligned locations
~~~~~~~~~~~~~~~~~~~~~~~
In the case where multiple consecutive sections are obtained and aligned with
``STIM``, the aligned locations can be obtained with::

    aligned_locations = sw.get_aligned_locations(pucks, puck_name,
                                                     transformation='model_sift')
