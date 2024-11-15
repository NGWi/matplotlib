"""
Functions for parsing scatter plot color arguments with support for edge color mapping.
"""

import numpy as np

import matplotlib.colors as mcolors
import matplotlib.cbook as cbook


def parse_scatter_color_args(c=None, edgecolors=None, ec=None, kwargs={},
                           xsize=0, get_next_color_func=None):
    """
    Process color specifications for scatter plot face and edge colors.

    Parameters
    ----------
    c : color or sequence of colors, optional
        Face colors. Can be:
        - A scalar or sequence of n numbers to be mapped to colors
        - A sequence of colors of length n
        - A single color format string

    edgecolors : color or sequence of colors, optional
        Edge colors specification using standard Matplotlib formats.
        If 'face', the edge color will be the same as the face color.

    ec : color or sequence of colors, optional
        Edge colors with enhanced support for color mapping. Can be:
        - A scalar or sequence of n numbers to be mapped to colors
        - A sequence of colors of length n
        - A single color format string
        Takes precedence over edgecolors if both are specified.

    kwargs : dict
        Additional keyword arguments that might contain color specifications.

    xsize : int
        Length of data being plotted, used for validation.

    get_next_color_func : callable
        Function to get the next color from the cycle.

    Returns
    -------
    c : array or None
        Processed face color values for color mapping.
    face_colors : array or None
        Processed face colors if not using color mapping.
    ec : array or None
        Processed edge color values for color mapping.
    edge_colors : array or None
        Processed edge colors if not using color mapping.

    Notes
    -----
    When ec is specified as an array of values, it can be mapped to colors using
    either ec_map/ec_norm/ec_vmin/ec_vmax parameters, or if those are not specified,
    it will use the face color parameters (cmap/norm/vmin/vmax).
    """
    # Handle face colors first (similar to original implementation)
    facecolors = kwargs.pop('facecolors', None)
    facecolors = kwargs.pop('facecolor', facecolors)
    
    # Handle edge colors
    edgecolors = kwargs.pop('edgecolor', edgecolors)
    
    # Get color from kwargs
    kwcolor = kwargs.pop('color', None)
    
    # Validate color arguments
    if kwcolor is not None and c is not None:
        raise ValueError("Supply a 'c' argument or a 'color' kwarg but not both; "
                        "they differ but their functionalities overlap.")
    
    # Set default colors based on kwcolor
    if kwcolor is not None:
        try:
            mcolors.to_rgba_array(kwcolor)
        except ValueError as err:
            raise ValueError(
                "'color' kwarg must be a color or sequence of color specs. "
                "For a sequence of values to be color-mapped, use the 'c' "
                "argument instead.") from err
        if edgecolors is None and ec is None:
            edgecolors = kwcolor
        if facecolors is None:
            facecolors = kwcolor
    
    # Process face colors
    c_was_none = c is None
    if c is None:
        c = (facecolors if facecolors is not None
             else get_next_color_func())
    
    # Process edge colors
    if ec is not None:
        # ec takes precedence over edgecolors
        edge_c = ec
        edge_is_mapped = True
    else:
        edge_c = edgecolors
        edge_is_mapped = False
    
    # Convert face colors
    face_colors = None
    if not c_was_none:
        try:
            c = np.asanyarray(c, dtype=float)
            face_is_mapped = True
        except (TypeError, ValueError):
            face_is_mapped = False
    
    if not face_is_mapped:
        try:
            face_colors = mcolors.to_rgba_array(c)
            if len(face_colors) not in (0, 1, xsize):
                raise ValueError(
                    f"'c' argument has {len(face_colors)} elements, which is "
                    f"inconsistent with x and y with size {xsize}")
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"'c' argument must be a color, a sequence of colors, or "
                f"a sequence of numbers, not {c!r}") from err
    
    # Convert edge colors if numeric values provided
    edge_colors = None
    if edge_is_mapped:
        try:
            edge_c = np.asanyarray(edge_c, dtype=float)
            if edge_c.size != xsize:
                raise ValueError(
                    f"'ec' argument has {edge_c.size} elements, which is "
                    f"inconsistent with x and y with size {xsize}")
        except (TypeError, ValueError):
            try:
                edge_colors = mcolors.to_rgba_array(edge_c)
                if len(edge_colors) not in (0, 1, xsize):
                    raise ValueError(
                        f"'ec' argument has {len(edge_colors)} elements, which is "
                        f"inconsistent with x and y with size {xsize}")
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"'ec' argument must be a color, a sequence of colors, or "
                    f"a sequence of numbers, not {edge_c!r}") from err
    elif edge_c is not None:
        try:
            edge_colors = mcolors.to_rgba_array(edge_c)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"'edgecolors' argument must be a color or sequence of colors, "
                f"not {edge_c!r}") from err
    
    return c, face_colors, edge_c, edge_colors
