# ruff: noqa: F401

from .array_utils import split_array
from .cmp import read_cmp, remove_cmp_formatting
from .df_utils import concat_TDs, copy_fields, rename_fields
from .dim_reduction import apply_dim_reduce_model, dim_reduce, fit_dim_reduce_model
from .extract_signals import (
    concat_trials,
    get_sig_by_trial,
    get_signals,
    reverse_concat,
    stack_time_average,
)
from .firing_rates import (
    add_firing_rates,
    get_average_firing_rates,
    remove_low_firing_neurons,
)
from .integrity_checks import all_integer, trials_are_same_length
from .interval import (
    extract_interval_from_signal,
    generate_epoch_fun,
    restrict_to_interval,
    slice_around_index,
    slice_around_point,
    slice_between_points,
    slice_in_trial,
)
from .io import mat2dataframe
from .movement_onset_and_peak import (
    add_movement_onset,
    add_peak_speed,
    add_peak_speed_idx,
    get_movement_onset,
    get_onset_idx,
    get_peak_speed,
    get_peak_speed_idx,
)
from .regression import (
    apply_regressor_model,
    expand_field_in_time,
    fit_regressor_model,
    regress,
)
from .signal_transformations import (
    center,
    center_normalize_signal,
    center_signal,
    get_range,
    soft_normalize_signal,
    sqrt_transform_signal,
    transform_signal,
    z_score,
    z_score_signal,
    zero_normalize_signal,
)
from .signals import add_gradient, add_norm, add_speed, signal_dimensionality
from .smoothing import hw_to_std, norm_gauss_window, smooth_data, smooth_signals
from .tools import (
    combine_time_bins,
    keep_common_trials,
    merge_signals,
    select_trials,
    subtract_cross_condition_mean,
    trial_average,
)
from .utils import (
    copy_td,
    determine_ref_field,
    get_array_fields,
    get_string_fields,
    get_time_varying_fields,
    get_trial_length,
    get_trial_lengths,
    remove_suffix,
)
