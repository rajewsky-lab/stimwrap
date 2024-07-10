import logging
import os

import h5py
import numpy as np
import zarr


class Dataset:
    """Handle dataset operations for both AnnData (.h5ad) and N5 (.n5) file
    formats."""

    def __init__(
        self, container: zarr.N5Store, dataset_name: str, mode: str = "r"
    ):
        """Initialize a Dataset object.

        Args:
            container (zarr.N5Store): The container storing the dataset.
            dataset_name (str): The name of the dataset.
            mode (str, optional): The mode to open the dataset. Defaults to "r" (read-only).

        Raises:
            ValueError: If the dataset name doesn't end with '.h5ad' or '.n5'.
        """

        self.container: zarr.N5Store = container
        "the root container"

        self.dataset_name: str = dataset_name
        "descriptive name of the dataset"

        self.file = None
        "either AnnData or n5 object"

        self.is_h5ad: bool = dataset_name.lower().endswith(".h5ad")
        self.is_n5: bool = dataset_name.lower().endswith(".n5")
        self.mode: str = mode
        "how the Dataset will be opened"

        if not (self.is_h5ad or self.is_n5):
            raise ValueError("The dataset name must end with '.h5ad' or '.n5'")

    def open(self):
        if self.is_h5ad:
            self.file = h5py.File(
                os.path.join(self.container.path, self.dataset_name), self.mode
            )
        elif self.is_n5:
            self.file = self.container.get(self.dataset_name)
            self.file.attrs = self.file._load_n5_attrs("attributes.json")

    def __enter__(self):
        self.open()
        return self

    def __str__(self, indent=""):
        strep = ""
        if self.is_h5ad:
            strep += f"{indent}stimwrap Dataset, Type: AnnData\n"
        elif self.is_n5:
            strep += f"{indent}stimwrap Dataset, Type: N5\n"

        return strep

    def __repr__(self):
        return self.__str__()

    def close(self):
        if self.is_h5ad and self.file:
            self.file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def cleanup_dataset(self):
        """Removes the housekeeping group `__DATA_TYPES__` created by JHDF5,
        which is kept if STIM is closed prematurely."""
        if self.is_h5ad and self.file:
            try:
                self.remove("__DATA_TYPES__")
            except KeyError as e:
                logging.warn(
                    f"Dataset {self.dataset_name} is clean (does not contain '__DATA_TYPES__'). Skipping..."
                )
            try:
                if not isinstance(self.file["obs"].attrs["column-order"], list):
                    self.file["obs"].attrs["column-order"] = ""
                if not isinstance(self.file["var"].attrs["column-order"], list):
                    self.file["var"].attrs["column-order"] = ""
            except KeyError as e:
                logging.warn(
                    f"Dataset {self.dataset_name} is clean (cannot reset 'column-order'). Skipping..."
                )
        else:
            logging.warn(
                f"Dataset {self.dataset_name} is not AnnData. Skipping..."
            )

    def get_attribute(self, attribute: str) -> np.ndarray:
        return np.array(self.file.attrs[attribute])

    def get(self, item: str) -> np.ndarray:
        """Get an item from the dataset.

        Args:
            item (str): The name of the item to retrieve.

        Returns:
            np.ndarray: The item value as a numpy array.
        """

        return np.array(self.file[item])

    def set(self, item: str, value: np.ndarray):
        """Set (write) an item in the dataset.

        Args:
            item (str): The name of the item to set.
            value (np.ndarray): The value to set.

        Raises:
            TypeError: If the value is not a numpy array.
            KeyError: If the item already exists in the dataset.
            ValueError: If the item cannot be created or written in the Dataset.
        """

        if not isinstance(value, np.ndarray):
            raise TypeError("The argument `value` must be a numpy array")
        if item in self.file:
            raise KeyError(f"The group '{item}' exists already")

        if self.is_h5ad and self.file:
            self.file[item] = value
        elif self.is_n5 and self.file:
            _newgrp = self.file.create_group(item)
            _newgrpvalue = _newgrp.zeros(item, shape=value.shape)
            _newgrpvalue[...] = value
        else:
            raise ValueError(
                f"'{item}' and values cannot be created/written in Dataset"
            )

    def __getitem__(self, key: str) -> np.ndarray:
        """Get an item from the dataset using square bracket notation.

        Args:
            key (str): The name of the item to retrieve.

        Returns:
            np.ndarray: The item value as a numpy array.

        Raises:
            KeyError: If the item is not found in the dataset.
        """
        try:
            return self.get(key)
        except KeyError:
            raise KeyError(f"'{key}' not found in the dataset")

    def __setitem__(self, key: str, value: np.ndarray):
        """Set an item in the dataset using square bracket notation.

        Args:
            key (str): The name of the item to set.
            value (np.ndarray): The value to set.

        Raises:
            TypeError: If the value is not a numpy array.
            KeyError: If the item already exists in the dataset.
            ValueError: If the item cannot be created or written in the Dataset.
        """
        self.set(key, value)

    def remove(self, item: str):
        """Remove an item from the dataset.

        Args:
            item (str): The name of the item to remove.
        """

        if self.is_h5ad and self.file:
            del self.file[item]
        else:
            logging.error(f"Cannot remove '{item}'")

    def get_attribute_or_item(self, item: str) -> np.ndarray:
        """Get an attribute or item from the dataset.

        Args:
            item (str): The name of the attribute or item to retrieve.

        Returns:
            np.ndarray: The attribute or item value as a numpy array.

        Raises:
            KeyError: If the item is not found in attributes or items.
        """

        if item in self.file.attrs:
            return self.get_attribute(item)
        elif item in self.file:
            return self.get(item)
        else:
            raise KeyError(f"Could not find '{item}' at attributes or items")

    def get_transform(self, transformation: str = "model_sift"):
        """Get the transformation matrix that is used to transform the original
        locations to the aligned locations.

        Args:
            transformation (str, optional): the transformation used. Can be
                'model_sift' (default), 'model_icp'.

        Returns:
            numpy array: the transform matrix.
        """
        if self.is_h5ad and self.file:
            transformation = (
                f"uns/{transformation}"
                if not transformation.startswith("uns/")
                else transformation
            )
            transform_matrix = self.get(transformation).reshape(2, 3)
        elif self.is_n5:
            transform_matrix = self.get_attribute(transformation)
        else:
            raise FileNotFoundError(
                "The dataset is not loaded or does not exist"
            )

        transform_matrix = np.concatenate(
            (transform_matrix, np.array([0, 0, 1]).reshape(1, 3))
        )
        return transform_matrix

    def apply_save_transform(
        self,
        transformation: str = "model_sift",
        locations: str = "locations",
        destination: str = "spatial_transform_sift",
        z_coord: float = None,
    ):
        """Get the transformation matrix that is used to transform the original
        locations to the aligned locations.

        If z_coord is provided, a 3D location vector is created per section (with the
        section Z-axis coordinate)

        Args:
            transformation (str, optional): the transformation used. Can be
                'model_sift' (default), 'model_icp'.
            locations (str, optional): the path where location data is found
            destination (str, optional): path where transformed coordinates
                will be stored (for AnnData, it is under 'obsm/')
            z_coord (float, optional): the Z-axis value for the section
        """
        locations = (
            f"obsm/{locations}"
            if not locations.startswith("obsm/") and self.is_h5ad
            else locations
        )
        aligned_locations = self.get_aligned_locations(
            transformation, locations
        )

        # append the provided z-axis to the transformed coordinates
        if z_coord is not None:
            transpose = (
                True
                if aligned_locations.shape[0] > aligned_locations.shape[1]
                else False
            )

            if transpose:
                aligned_locations = np.concatenate(
                    (
                        aligned_locations,
                        (np.ones(aligned_locations.shape[0]) * z_coord).reshape(
                            -1, 1
                        ),
                    ),
                    axis=1,
                )
            else:
                aligned_locations = np.concatenate(
                    (
                        aligned_locations,
                        (np.ones(aligned_locations.shape[1]) * z_coord).reshape(
                            1, -1
                        ),
                    ),
                    axis=0,
                )

        if self.is_h5ad and self.file and not destination.startswith("obsm/"):
            destination = f"obsm/{destination}"
        else:
            raise FileNotFoundError(
                "The dataset is not loaded or does not exist"
            )

        self.set(destination, aligned_locations)

    def get_aligned_locations(
        self, transformation: str = "model_sift", locations: str = "locations"
    ):
        """Get the aligned locations of a dataset after having aligned it to
        the rest datasets in the container.

        Args:.
            transformation (str, optional): the transformation used. Can be
                'model_sift' (default), 'model_icp'.
            locations (str, optional): the path where location data is found

        Returns:
            numpy array: the aligned locations.
        """
        locations = self.get(item=locations)

        # transpose flag
        transpose = False

        # we transpose if there location is at axis 0
        if locations.shape[1] < locations.shape[0]:
            locations = locations.T
            transpose = True

        # we proceed normally
        num_locations = locations.shape[1]
        locations = np.concatenate(
            (locations, np.ones(num_locations).reshape(1, num_locations))
        )
        transform_matrix = self.get_transform(transformation=transformation)
        aligned_locations = np.dot(transform_matrix, locations)[:2, :]

        if transpose:
            aligned_locations = aligned_locations.T

        return aligned_locations

    def get_gene_expression(self, gene: str = None):
        """Get gene expression from a specific dataset. It returns either a
        vector for a specific gene or the whole gene expression matrix if no
        gene is provided.

        Args:
            gene (str, optional): the gene to get the expression for. If  gene is None
                (default) the whole gene expression matrix is returned.

        Returns:
            numpy array: gene expression vector or matrix
        """
        if self.is_h5ad and self.file:
            if gene is not None:
                gene_idx = int(
                    np.where(np.array(self.file["var/_index"]) == gene)[0]
                )
            else:
                gene_idx = None

            _X = self.file["X"]
            _X_attrs = _X.attrs
            encoding_type = _X_attrs["encoding-type"]

            if encoding_type == "array":
                return self.file["X"][:, gene]
            elif encoding_type == "csr_matrix":
                _shape = _X_attrs["shape"]
                raise NotImplementedError(
                    "stimwrap cannot read expression from AnnData 'csr_matrix' yet."
                )
            elif encoding_type == "csc_matrix":
                _shape = _X_attrs["shape"]
                raise NotImplementedError(
                    "stimwrap cannot read expression from AnnData 'csc_matrix' yet."
                )
            else:
                raise ValueError(
                    "The gene expression stored in 'X' does not have compatible encoding"
                )
        elif self.is_n5:
            if gene is not None:
                gene_idx = int(
                    np.where(np.array(self.attrs["geneList"]) == gene)[0]
                )
                gene_expression = self.get("expression")[:, gene_idx]
                return gene_expression
            else:
                return np.round(np.array(self.get("expression")), 4)


class Container:
    """Parses the stores of various Datasets as a single N5 container."""
    def __init__(self, filename: str):
        self.path: str = filename
        "where the ``Container`` is stored"

        self.container: zarr.N5Store = zarr.N5Store(self.path)
        "the N5 Store"

        _attrs = self.container._load_n5_attrs("attributes.json")
        self.container.attrs = _attrs 
        self.attrs: dict = self.container.attrs
        "the root attributes of the container"

    def __str__(self):
        datasets = self.attrs["datasets"]
        num_datasets = self.attrs["num_datasets"]

        strep = f"stimwrap Container: {self.path}\n"
        strep += f"Number of datasets: {num_datasets}\n"
        strep += "Datasets:\n"

        for dataset in datasets:
            strep += f"  - {dataset}\n"
            with self.get_dataset(dataset) as ds:
                strep += ds.__str__(indent="    ")

        return strep

    def __repr__(self):
        return self.__str__()

    def cleanup_container(self):
        """Removes the housekeeping group `__DATA_TYPES__` created by JHDF5,
        which is kept if STIM is closed prematurely."""
        for dataset_name in self.container.attrs["datasets"]:
            with self.get_dataset(dataset_name, mode="r+") as dataset:
                dataset.cleanup_dataset()

    def get_container(filename: str):
        """Read the container as object.

        Args:
            filename (string): the path to the container.

        Returns:
            zarr.N5Store File: the container in zarr N5Store format.
        """

    def get_dataset_names(self):
        """Get all dataset names that are inside a container.

        Returns:
            list of strings: List of dataset names.
        """
        return self.container.attrs["datasets"]

    def get_dataset(self, dataset_name: str, mode: str = "r") -> Dataset:
        """Get a specific dataset from the container.

        This function supports both AnnData (.h5ad) and N5 (.n5) file formats.

        Args:
            dataset_name (str): the name of the dataset.

        Returns:
            Dataset: A Dataset object that can be used as a context manager.

        Raises:
            ValueError: if the dataset_name does not end with '.h5ad' or '.n5'.
        """
        return Dataset(self.container, dataset_name, mode)
