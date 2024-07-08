import os
import subprocess
import inspect
from functools import wraps

VALID_RENDERING_MODES = ["Gauss"]
VALID_STIM_VERSION = ["0.3.0"]
BIN_PATH = ""

# utils
def validate_file_exists(path: str) -> bool:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file or directory {path} does not exist.")
    return True

def validate_positive(value: float) -> bool:
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
    command = [os.path.join(BIN_PATH, 'st-help')] + ['--version']
    final_command = ' '.join(command)
    version = subprocess.run(final_command, shell=True, check=True, capture_output=True)
    return version.stdout.strip().decode()

# communicating with the Java bins
def check_version():
    import pkg_resources

    version = stim_version()
    stimwrap_version = "0.0.0"
    try:
        stimwrap_version = pkg_resources.get_distribution("stimwrap").version
    except pkg_resources.DistributionNotFound:
        pass

    if version not in VALID_STIM_VERSION:
        raise NotImplementedError(f"""This version of stimwrap ({stimwrap_version}) 
                                  is not compatible with STIM {version}.
                                  
                                  Please run stimwrap.set_bin_path("path_to_bins") to set
                                  the correct path (replace "path_to_bins" by the proper path)""")

def set_bin_path(default_path=None):
    global BIN_PATH
    conda_env = os.environ.get('CONDA_PREFIX')
    if default_path and os.path.exists(default_path):
        BIN_PATH = default_path
        return default_path
    elif conda_env:
        bins_path = os.path.join(conda_env, 'bin')
        if os.path.exists(bins_path):
            BIN_PATH = bins_path
            return bins_path
    raise FileNotFoundError("Binaries folder not found. Please specify a valid path.")

# general decorator to run command
def stim_function(program_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_version()
            args = func(*args, **kwargs)
            command = [os.path.join(BIN_PATH, program_name)] + args
            final_command = ' '.join(command)
            subprocess.run(final_command, shell=True, check=True)
        return wrapper
    return decorator

# data management
def add_slices(container: str,
               inputs: list,
               expression_values: str = None,
               locations: str = None,
               annotations: str = None,
               move: bool = False):
    
    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError("In `inputs`, you must provide the path to at least one file, as a list")
    
    for input in inputs:
        add_slice(container, input, expression_values, locations, annotations, move)


@stim_function("st-add-slice")
def add_slice(container: str,
              input: str,
              expression_values: str = None,
              locations: str = None,
              annotations: str = None,
              move: bool = False):

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
def resave(container: str,
           input: str,
           annotation: str = None,
           normalize: bool = False):

    validate_file_exists(input)

    
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
def normalize(container: str,
              input: str,
              output: str = None):
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
def add_annotations(input: str,
                    annotation: str,
                    label: str):
    validate_file_exists(input)
    validate_file_exists(annotation)

    args = [
        f"--input {input}",
        f"--annotation {annotation}",
        f"--label {label}"
    ]
    
    return args

@stim_function("st-add-entropy")
def add_entropy():
    pass

@stim_function("st-align-pairs-add")
def align_pairs_add(container: str,
                    datasets: list,
                    matches: str,
                    lmbda: float = 1.0,
                    scale: float = 0.05,
                    smoothness_factor: float = 4.0,
                    rendering_gene: str = None,
                    hide_pairwise_rendering: bool = False,
                    overwrite: bool = False):
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
def align_interactive(input: str,
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
        raise KeyError(f"The argument `rendering` must be one of {VALID_RENDERING_MODES}")

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
def align_pairs(container: str,
                datasets: list = None,
                genes: list = None,
                num_genes: int = 10,
                skip: int = 10,
                range: int = 2,
                max_epsilon: float = 2**32,
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
        raise KeyError(f"The argument `rendering` must be one of {VALID_RENDERING_MODES}")

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
    if overwrite:
        args.append("--overwrite")
    
    return args

@stim_function("st-align-global")
def align_global(container: str,
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
                ignore_quality: bool = False
                ):
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
def explorer(input: str,
             datasets: list = None):
    
    args = [
        f"--input {input}"
    ]

    if datasets:
        args.append(f"--datasets {','.join(datasets)}")

    return args

@stim_function("st-render")
def render(input: str,
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

    validate_file_exists(input)
    validate_positive(rendering_factor)
    validate_positive(scale)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(f"The argument `rendering` must be one of {VALID_RENDERING_MODES}")
    
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
def align_pairs_view(container: str,
                     datasets: list = None,
                     gene: str = None,
                     lmbda: float = 1.0,
                     scale: float = 0.05,
                     rendering_factor: float = 4.0):
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
def bdv_view3d(input: str,
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
    validate_file_exists(input)
    validate_positive(rendering_factor)
    validate_positive(z_spacing_factor)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(f"The argument `rendering` must be one of {VALID_RENDERING_MODES}")
    
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
def bdv_view(input: str,
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
             annotation_radius: float = 0.75
             ):
    validate_file_exists(input)
    validate_file_exists(dataset)
    validate_positive(rendering_factor)
    validate_positive(brightness_min)
    validate_positive(brightness_max)

    if rendering not in VALID_RENDERING_MODES:
        raise KeyError(f"The argument `rendering` must be one of {VALID_RENDERING_MODES}")
    
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