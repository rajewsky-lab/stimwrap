from ._stimwrap import Container, Dataset

from ._stim_functions import (
    add_slice,
    resave,
    normalize,
    add_annotations,
    add_entropy,
    align_pairs_add,
    align_interactive,
    align_pairs,
    align_global,
    explorer,
    align_pairs_view,
    bdv_view3d,
    bdv_view,
    stim_version,
    set_bin_path,
    add_slices,
)

# load the default bin path
set_bin_path()
