|PyPI|

.. |PyPI| image:: https://img.shields.io/pypi/v/stimwrap.svg
   :target: https://pypi.org/project/stimwrap/

stimwrap - a python interface for `STIM <https://github.com/PreibischLab/STIM/>`_
======================================

``stimwrap`` is a Python wrapper for the spatial transcriptomics library 
`STIM <https://github.com/PreibischLab/STIM/>`_. It provides an interface 
for running STIM commands from python, as well as for extracting datasets 
and their attributes from ``n5`` containers (backed by ``n5`` or ``AnnData``) 
that are created by STIM.

``stimwrap`` is created and maintained by `Nikos Karaiskos <mailto:nikolaos.karaiskos@mdc-berlin.de>`_
and `Daniel Leon-Perinan <mailto:daniel.leonperinan@mdc-berlin.de>`_.

Installation
------------
To install the ``stimwrap`` try::

    pip install stimwrap

or if you do not have sudo rights::

    pip install --user stimwrap

Check if the library is successfully installed::

    import stimwrap as sw

If installation fails due to conflicting dependencies, create a dedicated environment
with ``python>=3.7`` and try again to install ``stimwrap`` as above.

Usage
-----
The following assumes that the file ``container.n5`` contains the datasets and their
attributes as created by ``STIM``::

    pucks = sw.Container('/path/to/container.n5')

Print the names of the datasets::

    print(pucks.get_dataset_names())

Focus on a specific puck and extract the relevant information::

    puck_name = pucks.get_dataset_names()[0]
    puck = pucks.get_dataset(puck_name)

Get the puck locations either directly from the puck::

    locations = puck['locations']

or fetch them from the container::

    locations = pucks.get_dataset(puck_name)['locations']

Fetch gene expression
~~~~~~~~~~~~~~~~~~~~~
It is possible to get the expression vector of a single gene::

    hpca_vec = pucks.get_dataset(puck_name).get_gene_expression(gene='Hpca')

or the whole gene expression matrix::

    dge = pucks.get_dataset(puck_name).get_gene_expression()

Fetch dataset attributes
~~~~~~~~~~~~~~~~~~~~~~~~
``STIM`` stores the dataset attributes in the ``n5`` container. These can 
be directly accessed with ``stimwrap``::

    puck.get_attribute(attribute='geneList')

In N5-backed STIM, available options might also include: `barcodeList` and `metadataList`.

Fetch aligned locations
~~~~~~~~~~~~~~~~~~~~~~~
In the case where multiple consecutive sections are obtained and aligned with
``STIM``, the aligned locations can be obtained with::

    aligned_locations = puck.get_aligned_locations(transformation='model_sift')

Store aligned locations
~~~~~~~~~~~~~~~~~~~~~~~
The aligned locations can be stored in the N5 or AnnData-backed object, for
seamless downstream analysis::

    aligned_locations = puck.apply_save_transform(transformation='model_sift')