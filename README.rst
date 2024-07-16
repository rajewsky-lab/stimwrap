|PyPI|

.. |PyPI| image:: https://img.shields.io/pypi/v/stimwrap.svg
   :target: https://pypi.org/project/stimwrap/

stimwrap - a python interface for `STIM <https://github.com/PreibischLab/STIM/>`_
==================================================================================

``stimwrap`` is a Python wrapper for the spatial transcriptomics library 
`STIM <https://github.com/PreibischLab/STIM/>`_. It provides an interface 
for running STIM commands from python, as well as for extracting datasets 
and their attributes from ``n5`` containers (backed by ``n5`` or ``AnnData``) 
that are created by STIM.

``stimwrap`` is created and maintained by `Nikos Karaiskos <mailto:nikolaos.karaiskos@mdc-berlin.de>`_
and `Daniel León-Periñán <mailto:daniel.leonperinan@mdc-berlin.de>`_.

Quick start
-----------
We provide an example notebook under ``notebooks/openst_example.ipynb``, and also
a `Google Colab notebook <https://colab.research.google.com/drive/10j-pb0ZIC1OFIhRi1g5hkIeRTQQqEvL5?usp=sharing>`_

These will walk you through downloading `Open-ST data <https://rajewsky-lab.github.io/openst/latest/>`_, running
``STIM`` in a Python notebook (via ``stimwrap`` bindings), and seamlessly running some downstream analyses.

We provide another notebook under ``notebooks/visium_example.ipynb``, and also 
a `2nd Google Colab notebook <https://colab.research.google.com/drive/1dea0fbL1i65vYy3GvSc8RXX_mBGFia_O?usp=sharing>`_ 
showcasing the 10x Visium adult mouse brain dataset showcased in all our tutorials.

Installation
------------
To install the ``stimwrap`` try:

.. code-block:: bash

   pip install stimwrap

or if you do not have sudo rights:

.. code-block:: bash

   pip install --user stimwrap

Check if the library is successfully installed:

.. code-block:: bash

   python -c import stimwrap as sw

If installation fails due to conflicting dependencies, create a dedicated environment
with ``python>=3.9`` and try again to install ``stimwrap`` as above.

Usage
-----
The following assumes that the file ``container.n5`` contains the datasets and their
attributes as created by ``STIM``:

.. code-block:: python

   import stimwrap as sw
   pucks = sw.Container('/path/to/container.n5')

Print the names of the datasets:

.. code-block:: python
    
   print(pucks.get_dataset_names())

Focus on a specific puck and extract the relevant information:

.. code-block:: python
    
   puck_name = pucks.get_dataset_names()[0]
   puck = pucks.get_dataset(puck_name)
    
Get the puck locations either directly from the puck:

.. code-block:: python
    
   locations = puck['locations']

or fetch them from the container:

.. code-block:: python
    
   locations = pucks.get_dataset(puck_name)['locations']

The examples above assume that the dataset is `N5`-backed. For `AnnData`-backed
datasets, the key for the puck locations might be:

.. code-block:: python
    
   locations = pucks.get_dataset(puck_name)['spatial']

which will try to access the `obsm/spatial` variable from the dataset. Alternatively,
we recommend using the official `AnnData` package for handling these files.

Fetch gene expression
~~~~~~~~~~~~~~~~~~~~~
It is possible to get the expression vector of a single gene:

.. code-block:: python
    
   hpca_vec = pucks.get_dataset(puck_name).get_gene_expression(gene='Hpca')

or the whole gene expression matrix:

.. code-block:: python
    
   dge = pucks.get_dataset(puck_name).get_gene_expression()

Fetch dataset attributes
~~~~~~~~~~~~~~~~~~~~~~~~
``STIM`` stores the dataset attributes in the ``n5`` container. These can 
be directly accessed with ``stimwrap``:

.. code-block:: python
    
   puck.get_attribute(attribute='geneList')

In N5-backed STIM, available options might also include: `barcodeList` and `metadataList`.

Fetch aligned locations
~~~~~~~~~~~~~~~~~~~~~~~
In the case where multiple consecutive sections are obtained and aligned with
``STIM``, the aligned locations can be obtained with:

.. code-block:: python
    
   aligned_locations = puck.get_aligned_locations(transformation='model_sift')

Store aligned locations
~~~~~~~~~~~~~~~~~~~~~~~
The aligned locations can be stored in the N5 or AnnData-backed object, for
seamless downstream analysis:

.. code-block:: python
    
   aligned_locations = puck.apply_save_transform(transformation='model_sift')
