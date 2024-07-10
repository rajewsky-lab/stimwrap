import inspect
import logging
import os
import subprocess
import sys
from functools import wraps

VALID_RENDERING_MODES: list = ["Gauss"]
VALID_STIM_VERSION: list = ["0.3.0"]
BIN_PATH: str = ""


# utils
def validate_file_exists(path: str) -> bool:
    """Validates that a file or directory exists at the given path.

    Args:
        path (str): The path to the file or directory to check.

    Returns:
        bool: True if the file or directory exists.

    Raises:
        FileNotFoundError: If the file or directory does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file or directory {path} does not exist.")
    return True


def validate_positive(value: float) -> bool:
    """Validates that a given value is a non-negative number.

    This function checks if the input is a float or int and is greater than or equal to zero.
    It attempts to identify the variable name in the calling context for more informative error messages.

    Args:
        value (float): The value to validate.

    Returns:
        bool: True if the value is non-negative.

    Raises:
        ValueError: If the value is not a number or is negative.
    """
    if not isinstance(value, (float, int)) or value < 0:
        frame = inspect.currentframe().f_back
        variable_name = None
        for var_name, var_value in frame.f_locals.items():
            if var_value is value:
                variable_name = var_name
                break
        raise ValueError(f"{variable_name} must be non-negative.")
    return True


def stim_version():
    """Retrieves the version of the STIM software.

    This function calls the ``st-help`` command with the ``--version`` flag to get the STIM version.

    Returns:
        str: The version number of STIM as a string.

    Raises:
        subprocess.CalledProcessError: If the command fails to execute.
    """
    env = os.environ.copy()
    command = [os.path.join(BIN_PATH, "st-help")] + ["--version"]
    final_command = " ".join(command)
    version = subprocess.run(
        final_command, shell=True, check=True, capture_output=True, env=env
    )
    return version.stdout.strip().decode()


# communicating with the Java bins
def check_version():
    """Checks compatibility between the current stimwrap version and the
    installed STIM version.

    This function compares the installed STIM version with the list of valid STIM versions
    that are compatible with the current stimwrap version.

    Raises:
        NotImplementedError: If the installed STIM version is not compatible with the current stimwrap version.
    """
    import pkg_resources

    version = stim_version()
    stimwrap_version = "0.0.0"
    try:
        stimwrap_version = pkg_resources.get_distribution("stimwrap").version
    except pkg_resources.DistributionNotFound:
        pass

    if version not in VALID_STIM_VERSION:
        raise NotImplementedError(
            f"""This version of stimwrap ({stimwrap_version}) 
                                  is not compatible with STIM {version}.
                                  
                                  Please run stimwrap.set_bin_path("path_to_bins") to set
                                  the correct path (replace "path_to_bins" by the proper path)"""
        )


def set_bin_path(default_path=None):
    """Sets the global BIN_PATH variable to the location of STIM binaries.

    This function attempts to set the ``BIN_PATH`` in the following order:
    1. Uses the provided default_path if it exists.
    2. Checks for binaries in the current Conda environment, if one is active.
    3. Raises an error if neither option is successful.

    Args:
        default_path (str, optional): The path to the STIM binaries. If provided and valid, it will be used.

    Returns:
        str: The path to the STIM binaries that was successfully set.

    Raises:
        FileNotFoundError: If no valid binary path is found or provided.
    """
    global BIN_PATH
    conda_env = os.environ.get("CONDA_PREFIX")
    if default_path and os.path.exists(default_path):
        BIN_PATH = default_path
        return default_path
    elif conda_env:
        bins_path = os.path.join(conda_env, "bin")
        if os.path.exists(bins_path):
            BIN_PATH = bins_path
            return bins_path
    logging.warn(
        "Binaries folder not found. Please specify a valid path by running ``stimwrap.set_bin_path(...)``"
    )


# general decorator to run command
def stim_function(program_name):
    """Decorator for running a program under ``BIN_PATH``
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_version()
            args = func(*args, **kwargs)
            command = [os.path.join(BIN_PATH, program_name)] + args
            final_command = " ".join(command)
            env = os.environ.copy()
            process = subprocess.Popen(
                final_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
                sys.stdout.flush()

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        return wrapper

    return decorator


# data management
def add_slices(
    container: str,
    inputs: list,
    expression_values: str = None,
    locations: str = None,
    annotations: str = None,
    move: bool = False,
):
    """Adds multiple slices to a container.

    This function iterates over the input files and calls `stimwrap.add_slice` for each one.

    Args:
        container (str): Path to the container file.
        inputs (list): List of paths to input files to be added as slices.
        expression_values (str, optional): Path to expression values (in n5, or AnnData).
        locations (str, optional): Path to locations (in n5, or AnnData).
        annotations (str, optional): Path to annotations (in n5, or AnnData).
        move (bool, optional): If True, move the input files instead of copying. Default is False.

    Raises:
        ValueError: If inputs is not a list or is empty.
    """

    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError(
            "In `inputs`, you must provide the path to at least one file, as a list"
        )

    for input in inputs:
        add_slice(
            container, input, expression_values, locations, annotations, move
        )


@stim_function("st-add-slice")
def add_slice(
    container: str,
    input: str,
    expression_values: str = None,
    locations: str = None,
    annotations: str = None,
    move: bool = False,
):
    """Adds a single slice to a container.

    This function prepares the arguments for the ``st-add-slice`` command and executes it.

    Args:
        container (str): Path to the container file.
        input (str): Path to the input file to be added as a slice.
        expression_values (str, optional): Path to expression values (in n5, or AnnData).
        locations (str, optional): Path to locations (in n5, or AnnData).
        annotations (str, optional): Path to annotations (in n5, or AnnData).
        move (bool, optional): If True, move the input file instead of copying. Default is False.

    Returns:
    list: A list of command-line arguments for the ``st-add-slice`` command.
    """

    validate_file_exists(input)

    args = [
        f"--container {container}",
        f"--input {input}",
    ]

    if expression_values:
        args.append(f"--expressionValues {expression_values}")
    if locations:
        args.append(f"--locations {locations}")
    if annotations:
        args.append(f"--annotations {annotations}")
    if move:
        args.append("--move")

    return args


@stim_function("st-resave")
def resave(
    container: str, input: str, annotation: str = None, normalize: bool = False
):
    """Resaves data in a container.

    This function prepares the arguments for the ``st-resave`` command and executes it.

    Args:
        container (str): Path to the container file.
        input (str): Path to the input file.
        annotation (str, optional): Path to annotation file.
        normalize (bool, optional): If True, normalize the data. Default is False.

    Returns:
        list: A list of command-line arguments for the ``st-resave`` command.
    """

    args = [
        f"--container {container}",
        f"--input {input}",
    ]

    if annotation:
        args.append(f"--annotation {annotation}")
    if normalize:
        args.append("--normalize")

    return args


@stim_function("st-normalize")
def normalize(container: str, input: str, output: str = None):
    """Normalizes data in a container.

    This function prepares the arguments for the ``st-normalize`` command and executes it.

    Args:
        container (str): Path to the container file.
        input (str): Path to the input file.
        output (str, optional): Path to the output file.

    Returns:
        list: A list of command-line arguments for the ``st-normalize`` command.
    """

    validate_file_exists(container)
    validate_file_exists(input)

    args = [
        f"--container {container}",
        f"--input {input}",
    ]

    if output:
        args.append(f"--output {output}")

    return args


@stim_function("st-add-annotations")
def add_annotations(input: str, annotation: str, label: str):
    """Adds annotations to a container.

    This function prepares the arguments for the ``st-add-annotations`` command and executes it.

    Args:
        input (str): Path to the input file.
        annotation (str): Path to the annotation file.
        label (str): Label for the annotation.

    Returns:
        list: A list of command-line arguments for the ``st-add-annotations`` command.
    """

    validate_file_exists(input)
    validate_file_exists(annotation)

    args = [
        f"--input {input}",
        f"--annotation {annotation}",
        f"--label {label}",
    ]

    return args


@stim_function("st-add-entropy")
def add_entropy():
    pass


@stim_function("st-align-pairs-add")
def align_pairs_add(
    container: str,
    datasets: list,
    matches: str,
    lmbda: float = 1.0,
    scale: float = 0.05,
    smoothness_factor: float = 4.0,
    rendering_gene: str = None,
    hide_pairwise_rendering: bool = False,
    overwrite: bool = False,
):
    """Aligns pairs of datasets and adds the alignment to the container.

    This function prepares the arguments for the ``st-align-pairs-add`` command and executes it.

    Args:
        container (str): Path to the container file.
        datasets (list): List of datasets to align.
        matches (str): Path to the matches file.
        lmbda (float, optional): Lambda parameter for alignment. Default is 1.0.
        scale (float, optional): Scale parameter for alignment. Default is 0.05.
        smoothness_factor (float, optional): Smoothness factor for alignment. Default is 4.0.
        rendering_gene (str, optional): Gene to use for rendering.
        hide_pairwise_rendering (bool, optional): If True, hide pairwise rendering. Default is False.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.

    Returns:
        list: A list of command-line arguments for the ``st-align-pairs-add`` command.
    """
    validate_file_exists(container)
    validate_file_exists(matches)

    args = [
        f"--container {container}",
        f"--matches {matches}",
        f"--lambda {lmbda}",
        f"--scale {scale}",
        f"--smoothnessFactor {smoothness_factor}",
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")
    if rendering_gene:
        args.append(f"--renderingGene {rendering_gene}")
    if hide_pairwise_rendering:
        args.append("--hidePairwiseRendering")
    if overwrite:
        args.append("--overwrite")

    return args


# alignment
@stim_function("st-align-interactive")
def align_interactive(
    input: str,
    section_a: str,
    section_b: str,
    num_genes: int = 10,
    skip: int = 10,
    rendering: str = "Gauss",
    rendering_factor: float = 1.0,
    scale: float = 0.05,
    brightness_min: float = 0,
    brightness_max: float = 0.5,
    ff_gauss: float = None,
    ff_mean: float = None,
    ff_median: float = None,
    ff_single_spot: float = None,
):
    """Performs interactive alignment of two sections.

    This function prepares the arguments for the ``st-align-interactive`` command and executes it.

    Args:
        input (str): Path to the input file.
        section_a (str): Path to the first section file.
        section_b (str): Path to the second section file.
        num_genes (int, optional): Number of genes to use for alignment. Default is 10.
        skip (int, optional): Number of genes to skip. Default is 10.
        rendering (str, optional): Rendering mode. Default is "Gauss".
        rendering_factor (float, optional): Rendering factor. Default is 1.0.
        scale (float, optional): Scale factor. Default is 0.05.
        brightness_min (float, optional): Minimum brightness. Default is 0.
        brightness_max (float, optional): Maximum brightness. Default is 0.5.
        ff_gauss (float, optional): Gaussian filter factor.
        ff_mean (float, optional): Mean filter factor.
        ff_median (float, optional): Median filter factor.
        ff_single_spot (float, optional): Single spot filter factor.

    Returns:
        list: A list of command-line arguments for the ``st-align-interactive`` command.

    Raises:
        KeyError: If an invalid rendering mode is provided.
    """

    validate_file_exists(input)
    validate_file_exists(section_a)
    validate_file_exists(section_b)
    validate_positive(num_genes)
    validate_positive(skip)
    validate_positive(rendering_factor)
    validate_positive(scale)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(
            f"The argument `rendering` must be one of {VALID_RENDERING_MODES}"
        )

    if ff_gauss is not None:
        validate_positive(ff_gauss)
    if ff_mean is not None:
        validate_positive(ff_mean)
    if ff_median is not None:
        validate_positive(ff_median)
    if ff_single_spot is not None:
        validate_positive(ff_single_spot)

    args = [
        f"--input {input}",
        f"--dataset1 {section_a}",
        f"--dataset2 {section_b}",
        f"--numGenes {num_genes}",
        f"--skip {skip}",
        f"--rendering {rendering}",
        f"--renderingFactor {rendering_factor}",
        f"--scale {scale}",
        f"--brightnessMin {brightness_min}",
        f"--brightnessMax {brightness_max}",
    ]

    if ff_gauss is not None:
        args.append(f"--ffGauss {ff_gauss}")
    if ff_mean is not None:
        args.append(f"--ffMean {ff_mean}")
    if ff_median is not None:
        args.append(f"--ffMedian {ff_median}")
    if ff_single_spot is not None:
        args.append(f"--ffSingleSpot {ff_single_spot}")

    return args


@stim_function("st-align-pairs")
def align_pairs(
    container: str,
    datasets: list = None,
    genes: list = None,
    num_genes: int = 10,
    skip: int = 10,
    entropy_path: str = None,
    range: int = 2,
    max_epsilon: float = 0,
    min_num_inliers: int = 30,
    min_num_inliers_gene: int = 5,
    rendering: str = "Gauss",
    rendering_factor: float = 1.0,
    scale: float = 0.05,
    brightness_min: float = 0,
    brightness_max: float = 0.5,
    ff_gauss: float = None,
    ff_mean: float = None,
    ff_median: float = None,
    ff_single_spot: float = None,
    hide_pairwise_rendering: bool = True,
    overwrite: bool = False,
    num_threads: int = 0,
):
    """Aligns pairs of datasets.

    This function prepares the arguments for the ``st-align-pairs`` command and executes it.

    Args:
        container (str): Path to the container file.
        datasets (list, optional): List of datasets to align.
        genes (list, optional): List of genes to use for alignment.
        num_genes (int, optional): Number of genes to use if genes list is not provided. Default is 10.
        entropy_path (str, optional): If precomputed, where the gene variability metric is found.
        skip (int, optional): Number of genes to skip. Default is 10.
        range (int, optional): Range parameter. Default is 2.
        max_epsilon (float, optional): Maximum epsilon value. Default is 0 (will be automatically computed by STIM).
        min_num_inliers (int, optional): Minimum number of inliers. Default is 30.
        min_num_inliers_gene (int, optional): Minimum number of inliers per gene. Default is 5.
        rendering (str, optional): Rendering mode. Default is "Gauss".
        rendering_factor (float, optional): Rendering factor. Default is 1.0.
        scale (float, optional): Scale factor. Default is 0.05.
        brightness_min (float, optional): Minimum brightness. Default is 0.
        brightness_max (float, optional): Maximum brightness. Default is 0.5.
        ff_gauss (float, optional): Gaussian filter factor.
        ff_mean (float, optional): Mean filter factor.
        ff_median (float, optional): Median filter factor.
        ff_single_spot (float, optional): Single spot filter factor.
        hide_pairwise_rendering (bool, optional): If True, hide pairwise rendering. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
        num_threads (int, optional): Number of threads to use. Default is 0 (use all available).

    Returns:
        list: A list of command-line arguments for the ``st-align-pairs`` command.

    Raises:
        KeyError: If an invalid rendering mode is provided.
    """

    validate_file_exists(container)
    validate_positive(num_genes)
    validate_positive(skip)
    validate_positive(range)
    validate_positive(max_epsilon)
    validate_positive(min_num_inliers)
    validate_positive(min_num_inliers_gene)
    validate_positive(rendering_factor)
    validate_positive(scale)
    validate_positive(brightness_min)
    validate_positive(brightness_max)
    validate_positive(num_threads)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(
            f"The argument `rendering` must be one of {VALID_RENDERING_MODES}"
        )

    args = [
        f"--container {container}",
        f"--skip {skip}",
        f"--range {range}",
        f"--maxEpsilon {max_epsilon}",
        f"--minNumInliers {min_num_inliers}",
        f"--minNumInliersGene {min_num_inliers_gene}",
        f"--rendering {rendering}",
        f"--renderingFactor {rendering_factor}",
        f"--scale {scale}",
        f"--brightnessMin {brightness_min}",
        f"--brightnessMax {brightness_max}",
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")
    if genes:
        args.append(f"--genes {','.join(genes)}")
        args.append(f"--numGenes 0")
    else:
        args.append(f"--numGenes {num_genes}")
    if ff_gauss is not None:
        args.append(f"--ffGauss {ff_gauss}")
    if ff_mean is not None:
        args.append(f"--ffMean {ff_mean}")
    if ff_median is not None:
        args.append(f"--ffMedian {ff_median}")
    if ff_single_spot is not None:
        args.append(f"--ffSingleSpot {ff_single_spot}")
    if hide_pairwise_rendering:
        args.append("--hidePairwiseRendering")
    if num_threads > 0:
        args.append(f"--numThreads {num_threads}")
    elif num_threads == 0:
        args.append(f"--numThreads {os.cpu_count()}")
    if entropy_path is not None:
        args.append(f"--entropyPath {entropy_path}")
    if overwrite:
        args.append("--overwrite")

    return args


@stim_function("st-align-global")
def align_global(
    container: str,
    datasets: list = None,
    absolute_threshold: float = 160.0,
    lmbda: float = 1.0,
    icp_error_fraction: float = 1.0,
    icp_iterations: int = 100,
    min_iterations: int = 500,
    max_iterations: int = 3000,
    min_iterations_icp: int = 500,
    max_iterations_icp: int = 3000,
    relative_threshold: float = 3.0,
    rendering_factor: float = 1.0,
    display_gene: str = None,
    skip_display_results: bool = True,
    skip_icp: bool = False,
    ignore_quality: bool = False,
):
    """Performs global alignment of datasets.

    This function prepares the arguments for the ``st-align-global`` command and executes it.

    Args:
        container (str): Path to the container file.
        datasets (list, optional): List of datasets to align.
        absolute_threshold (float, optional): Absolute threshold for alignment. Default is 160.0.
        lmbda (float, optional): Lambda parameter for alignment. Default is 1.0.
        icp_error_fraction (float, optional): ICP error fraction. Default is 1.0.
        icp_iterations (int, optional): Number of ICP iterations. Default is 100.
        min_iterations (int, optional): Minimum number of iterations. Default is 500.
        max_iterations (int, optional): Maximum number of iterations. Default is 3000.
        min_iterations_icp (int, optional): Minimum number of ICP iterations. Default is 500.
        max_iterations_icp (int, optional): Maximum number of ICP iterations. Default is 3000.
        relative_threshold (float, optional): Relative threshold for alignment. Default is 3.0.
        rendering_factor (float, optional): Rendering factor. Default is 1.0.
        display_gene (str, optional): Gene to display in results.
        skip_display_results (bool, optional): If True, skip displaying results. Default is True.
        skip_icp (bool, optional): If True, skip ICP alignment. Default is False.
        ignore_quality (bool, optional): If True, ignore quality checks. Default is False.

    Returns:
        list: A list of command-line arguments for the ``st-align-global`` command.
    """

    validate_file_exists(container)
    validate_positive(absolute_threshold)
    validate_positive(lmbda)
    validate_positive(icp_error_fraction)
    validate_positive(icp_iterations)
    validate_positive(min_iterations)
    validate_positive(max_iterations)
    validate_positive(min_iterations_icp)
    validate_positive(max_iterations_icp)
    validate_positive(relative_threshold)
    validate_positive(rendering_factor)

    args = [
        f"--container {container}",
        f"--absoluteThreshold {absolute_threshold}",
        f"--lambda {lmbda}",
        f"--icpErrorFraction {icp_error_fraction}",
        f"--icpIterations {icp_iterations}",
        f"--minIterations {min_iterations}",
        f"--maxIterations {max_iterations}",
        f"--minIterationsICP {min_iterations_icp}",
        f"--maxIterationsICP {max_iterations_icp}",
        f"--relativeThreshold {relative_threshold}",
        f"--renderingFactor {rendering_factor}",
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")
    if display_gene:
        args.append(f"--gene {display_gene}")
    if skip_display_results:
        args.append("--skipDisplayResults")
    if skip_icp:
        args.append("--skipICP")
    if ignore_quality:
        args.append("--ignoreQuality")

    return args


# interactive exploration/visualization
@stim_function("st-explorer")
def explorer(input: str, datasets: list = None):
    """Launches the STIM explorer for interactive data exploration.

    This function prepares the arguments for the ``st-explorer`` command and executes it.

    Args:
        input (str): Path to the input file.
        datasets (list, optional): List of datasets to explore.

    Returns:
        list: A list of command-line arguments for the ``st-explorer`` command.
    """
    args = [f"--input {input}"]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")

    return args


@stim_function("st-render")
def render(
    input: str,
    output: str,
    datasets: list = None,
    genes: list = None,
    rendering: str = "Gauss",
    rendering_factor: float = 1.0,
    scale: float = 0.05,
    border: int = 20,
    brightness_min: float = 0,
    brightness_max: float = 0.5,
    ff_gauss: float = None,
    ff_mean: float = None,
    ff_median: float = None,
    ff_single_spot: float = None,
    ignore_transforms: bool = False,
):
    """Renders datasets and genes.

    This function prepares the arguments for the ``st-render`` command and executes it.

    Args:
        input (str): Path to the input file.
        output (str): Path to the output file.
        datasets (list, optional): List of datasets to render.
        genes (list, optional): List of genes to render.
        rendering (str, optional): Rendering mode. Default is "Gauss".
        rendering_factor (float, optional): Rendering factor. Default is 1.0.
        scale (float, optional): Scale factor. Default is 0.05.
        border (int, optional): Border size. Default is 20.
        brightness_min (float, optional): Minimum brightness. Default is 0.
        brightness_max (float, optional): Maximum brightness. Default is 0.5.
        ff_gauss (float, optional): Gaussian filter factor.
        ff_mean (float, optional): Mean filter factor.
        ff_median (float, optional): Median filter factor.
        ff_single_spot (float, optional): Single spot filter factor.
        ignore_transforms (bool, optional): If True, ignore transforms. Default is False.

    Returns:
        list: A list of command-line arguments for the ``st-render`` command.

    Raises:
        KeyError: If an invalid rendering mode is provided.
    """

    validate_file_exists(input)
    validate_positive(rendering_factor)
    validate_positive(scale)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(
            f"The argument `rendering` must be one of {VALID_RENDERING_MODES}"
        )

    if ff_gauss is not None:
        validate_positive(ff_gauss)
    if ff_mean is not None:
        validate_positive(ff_mean)
    if ff_median is not None:
        validate_positive(ff_median)
    if ff_single_spot is not None:
        validate_positive(ff_single_spot)

    args = [
        f"--input {input}",
        f"--output {output}",
        f"--rendering {rendering}",
        f"--renderingFactor {rendering_factor}",
        f"--scale {scale}",
        f"--border {border}",
        f"--brightnessMin {brightness_min}",
        f"--brightnessMax {brightness_max}",
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")
    if genes:
        args.append(f"--genes {','.join(genes)}")
    if ff_gauss is not None:
        args.append(f"--ffGauss {ff_gauss}")
    if ff_mean is not None:
        args.append(f"--ffMean {ff_mean}")
    if ff_median is not None:
        args.append(f"--ffMedian {ff_median}")
    if ff_single_spot is not None:
        args.append(f"--ffSingleSpot {ff_single_spot}")
    if ignore_transforms:
        args.append("--ignoreTransforms")

    return args


@stim_function("st-align-pairs-view")
def align_pairs_view(
    container: str,
    datasets: list = None,
    gene: str = None,
    lmbda: float = 1.0,
    scale: float = 0.05,
    rendering_factor: float = 4.0,
):
    """Views the alignment of pairs of datasets.

    This function prepares the arguments for the ``st-align-pairs-view`` command and executes it.

    Args:
        container (str): Path to the container file.
        datasets (list, optional): List of datasets to view.
        gene (str, optional): Gene to use for visualization.
        lmbda (float, optional): Lambda parameter for alignment. Default is 1.0.
        scale (float, optional): Scale factor. Default is 0.05.
        rendering_factor (float, optional): Rendering factor. Default is 4.0.

    Returns:
        list: A list of command-line arguments for the ``st-align-pairs-view`` command.
    """

    validate_file_exists(container)
    validate_positive(lmbda)
    validate_positive(scale)
    validate_positive(rendering_factor)

    args = [
        f"--container {container}",
        f"--lambda {lmbda}",
        f"--scale {scale}",
        f"--renderingFactor {rendering_factor}",
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")
    if gene:
        args.append(f"--gene {gene}")

    return args


@stim_function("st-bdv-view3d")
def bdv_view3d(
    input: str,
    genes: list,
    datasets: list = None,
    annotation: str = None,
    rendering: str = "Gauss",
    rendering_factor: float = 1.0,
    z_spacing_factor: float = 10.0,
    brightness_min: float = 0,
    brightness_max: float = 0.5,
    ff_gauss: float = None,
    ff_mean: float = None,
    ff_median: float = None,
    ff_single_spot: float = None,
    annotation_radius: float = 0.75,
):
    """Launches a 3D viewer for the datasets.

    This function prepares the arguments for the ``st-bdv-view3d`` command and executes it.

    Args:
        input (str): Path to the input file.
        genes (list): List of genes to visualize.
        datasets (list, optional): List of datasets to visualize.
        annotation (str, optional): Path to annotation file.
        rendering (str, optional): Rendering mode. Default is "Gauss".
        rendering_factor (float, optional): Rendering factor. Default is 1.0.
        z_spacing_factor (float, optional): Z-spacing factor. Default is 10.0.
        brightness_min (float, optional): Minimum brightness. Default is 0.
        brightness_max (float, optional): Maximum brightness. Default is 0.5.
        ff_gauss (float, optional): Gaussian filter factor.
        ff_mean (float, optional): Mean filter factor.
        ff_median (float, optional): Median filter factor.
        ff_single_spot (float, optional): Single spot filter factor.
        annotation_radius (float, optional): Annotation radius. Default is 0.75.

    Returns:
        list: A list of command-line arguments for the ``st-bdv-view3d`` command.

    Raises:
        KeyError: If an invalid rendering mode is provided.
    """

    validate_file_exists(input)
    validate_positive(rendering_factor)
    validate_positive(z_spacing_factor)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(
            f"The argument `rendering` must be one of {VALID_RENDERING_MODES}"
        )

    if ff_gauss is not None:
        validate_positive(ff_gauss)
    if ff_mean is not None:
        validate_positive(ff_mean)
    if ff_median is not None:
        validate_positive(ff_median)
    if ff_single_spot is not None:
        validate_positive(ff_single_spot)
    validate_positive(annotation_radius)

    args = [
        f"--input {input}",
        f"--genes {','.join(genes)}",
        f"--rendering {rendering}",
        f"--renderingFactor {rendering_factor}",
        f"--zSpacingFactor {z_spacing_factor}",
        f"--brightnessMin {brightness_min}",
        f"--brightnessMax {brightness_max}",
        f"--annotationRadius {annotation_radius}",
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")
    if annotation:
        args.append(f"--annotation {annotation}")
    if ff_gauss is not None:
        args.append(f"--ffGauss {ff_gauss}")
    if ff_mean is not None:
        args.append(f"--ffMean {ff_mean}")
    if ff_median is not None:
        args.append(f"--ffMedian {ff_median}")
    if ff_single_spot is not None:
        args.append(f"--ffSingleSpot {ff_single_spot}")

    return args


@stim_function("st-bdv-view")
def bdv_view(
    input: str,
    genes: list,
    dataset: str,
    annotation: str = None,
    rendering: str = "Gauss",
    rendering_factor: float = 1.0,
    brightness_min: float = 0,
    brightness_max: float = 0.5,
    ff_gauss: float = None,
    ff_mean: float = None,
    ff_median: float = None,
    ff_single_spot: float = None,
    annotation_radius: float = 0.75,
):
    """Launches a 2D viewer for a single dataset.

    This function prepares the arguments for the ``st-bdv-view`` command and executes it.

    Args:
        input (str): Path to the input file.
        genes (list): List of genes to visualize.
        dataset (str): Dataset to visualize.
        annotation (str, optional): Path to annotation file.
        rendering (str, optional): Rendering mode. Default is "Gauss".
        rendering_factor (float, optional): Rendering factor. Default is 1.0.
        brightness_min (float, optional): Minimum brightness. Default is 0.
        brightness_max (float, optional): Maximum brightness. Default is 0.5.
        ff_gauss (float, optional): Gaussian filter factor.
        ff_mean (float, optional): Mean filter factor.
        ff_median (float, optional): Median filter factor.
        ff_single_spot (float, optional): Single spot filter factor.
        annotation_radius (float, optional): Annotation radius. Default is 0.75.

    Returns:
        list: A list of command-line arguments for the ``st-bdv-view`` command.

    Raises:
        KeyError: If an invalid rendering mode is provided.
    """

    validate_file_exists(input)
    validate_file_exists(dataset)
    validate_positive(rendering_factor)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(
            f"The argument `rendering` must be one of {VALID_RENDERING_MODES}"
        )

    if ff_gauss is not None:
        validate_positive(ff_gauss)
    if ff_mean is not None:
        validate_positive(ff_mean)
    if ff_median is not None:
        validate_positive(ff_median)
    if ff_single_spot is not None:
        validate_positive(ff_single_spot)
    validate_positive(annotation_radius)

    args = [
        f"--input {input}",
        f"--genes {','.join(genes)}",
        f"--dataset {dataset}",
        f"--rendering {rendering}",
        f"--renderingFactor {rendering_factor}",
        f"--brightnessMin {brightness_min}",
        f"--brightnessMax {brightness_max}",
        f"--annotationRadius {annotation_radius}",
    ]

    if annotation:
        args.append(f"--annotation {annotation}")
    if ff_gauss is not None:
        args.append(f"--ffGauss {ff_gauss}")
    if ff_mean is not None:
        args.append(f"--ffMean {ff_mean}")
    if ff_median is not None:
        args.append(f"--ffMedian {ff_median}")
    if ff_single_spot is not None:
        args.append(f"--ffSingleSpot {ff_single_spot}")

    return args
