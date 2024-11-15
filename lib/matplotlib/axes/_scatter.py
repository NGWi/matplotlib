"""
Enhanced scatter plot implementation with support for edge color mapping.
"""

import numpy as np

from matplotlib import _api, colors as mcolors, collections
from matplotlib.axes._scatter_edge_colormap import parse_scatter_color_args


def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None,
           vmin=None, vmax=None, alpha=None, linewidths=None, *,
           edgecolors=None, ec=None, ec_map=None, ec_norm=None,
           ec_vmin=None, ec_vmax=None, plotnonfinite=False, **kwargs):
    """
    A scatter plot of y vs. x with varying marker size and/or color.

    Parameters
    ----------
    x, y : float or array-like, shape (n, )
        The data positions.

    s : float or array-like, shape (n, ), optional
        The marker size in points**2.
        Default is ``rcParams['lines.markersize'] ** 2``.

    c : array-like or list of colors or color, optional
        The marker face colors. Possible values:

        - A scalar or sequence of n numbers that will be mapped to colors using
          *cmap* and *norm*.
        - A 2D array in which the rows are RGB or RGBA.
        - A sequence of colors of length n.
        - A single color format string.

        Note that *c* should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values to be
        colormapped. If you want to specify the same RGB or RGBA value for
        all points, use a 2D array with a single row.

    marker : `~.markers.MarkerStyle` or ``{'.'|','|'o'|'v'|'^'|...}``, default: ','
        The marker style. *marker* can be either an instance of the class or
        the text shorthand for a particular marker.
        See :mod:`matplotlib.markers` for more information about marker styles.

    cmap : str or `~matplotlib.colors.Colormap`, optional
        A `.Colormap` instance or registered colormap name. *cmap* is only used
        if *c* is an array of floats.

    norm : `~matplotlib.colors.Normalize`, optional
        If *c* is an array of floats, *norm* is used to scale the color data,
        *c*, in the range 0 to 1, in order to map into the colormap *cmap*.
        If *None*, use the default `.colors.Normalize`.

    vmin, vmax : float, optional
        *vmin* and *vmax* are used in conjunction with the default norm to
        map the color array *c* to the colormap *cmap*. If None, the
        respective min and max of the color array is used.
        It is deprecated to use *vmin*/*vmax* when *norm* is given.

    alpha : float or array-like, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque).

    linewidths : float or array-like, optional
        The linewidth of the marker edges. Note: The default *edgecolors*
        is 'face'. You may want to change this as well.
        If *None*, defaults to rcParams ``lines.linewidth``.

    edgecolors : {'face', 'none', *None*} or color or sequence of color, optional
        The edge color of the marker. Possible values:

        - 'face': The edge color will always be the same as the face color.
        - 'none': No patch boundary will be drawn.
        - A color or sequence of colors.

        For non-filled markers, the *edgecolors* kwarg is ignored and forced
        to 'face' internally.

    ec : array-like or list of colors or color, optional
        The marker edge colors. Supports the same formats as 'c'.
        Takes precedence over edgecolors if both are specified.

    ec_map : str or `~matplotlib.colors.Colormap`, optional
        A `.Colormap` instance or registered colormap name for edge colors.
        If not specified and *ec* is an array of floats, defaults to *cmap*.

    ec_norm : `~matplotlib.colors.Normalize`, optional
        Normalization for edge colors. If not specified and *ec* is an array
        of floats, defaults to *norm*.

    ec_vmin, ec_vmax : float, optional
        *vmin* and *vmax* for edge colors. If not specified and *ec* is an
        array of floats, defaults to *vmin* and *vmax* respectively.

    plotnonfinite : bool, default: False
        Whether to plot points with nonfinite *c* (i.e. ``inf``, ``-inf`` or
        ``nan``). If ``True`` the points are drawn with the *bad* colormap color
        (see `.Colormap.set_bad`).

    Returns
    -------
    `~matplotlib.collections.PathCollection`

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.collections.Collection` properties

    See Also
    --------
    plot : To plot scatter plots when markers are identical in size and color.

    Notes
    -----
    * The `.plot` function will be faster for scatterplots where markers
      don't vary in size or color.

    * Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in which
      case all masks will be combined and only unmasked points will be
      plotted.

    * If your scatter markers require multiple colors, you can create a
      `~matplotlib.collections.PathCollection` directly with a specific
      *facecolor*/*edgecolor*.
    """
    # Process inputs
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    if s is None:
        s = (20 if marker == ',' else 20)**2
    s = np.asarray(s)
    
    # Validate norm and vmin/vmax combinations
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError(
            "Passing a Normalize instance simultaneously with "
            "vmin/vmax is not supported. Please pass vmin/vmax "
            "directly to the norm when creating it.")
            
    if ec_norm is not None and (ec_vmin is not None or ec_vmax is not None):
        raise ValueError(
            "Passing a Normalize instance simultaneously with "
            "ec_vmin/ec_vmax is not supported. Please pass vmin/vmax "
            "directly to the norm when creating it.")
    
    # Get face and edge colors
    c, face_colors, ec, edge_colors = parse_scatter_color_args(
        c=c, edgecolors=edgecolors, ec=ec, kwargs=kwargs,
        xsize=x.size, get_next_color_func=self._get_patches_for_fill.get_next_color)
    
    # Create scatter collection
    collection = collections.PathCollection(
        (self._get_patches_for_fill.get_path(marker),),
        scales=(np.sqrt(s),),
        offsets=np.column_stack([x, y]),
        transOffset=self.transData,
        alpha=alpha,
        linewidths=linewidths)
    
    # Set face colors
    if face_colors is None:
        collection.set_array(c)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
    else:
        collection.set_facecolor(face_colors)
    
    # Set edge colors - use face color parameters if edge parameters not specified
    if edge_colors is None and ec is not None:
        # Create a separate collection for edge colors
        edge_collection = collections.PathCollection(
            (self._get_patches_for_fill.get_path(marker),),
            scales=(np.sqrt(s),),
            offsets=np.column_stack([x, y]),
            transOffset=self.transData,
            facecolor='none',
            linewidths=linewidths)
        edge_collection.set_array(ec)
        
        # Use face color parameters if edge parameters not specified
        edge_collection.set_cmap(ec_map if ec_map is not None else cmap)
        edge_collection.set_norm(ec_norm if ec_norm is not None else norm)
        
        # Set color limits based on edge or face parameters
        if ec_vmin is not None or ec_vmax is not None:
            edge_collection.set_clim(ec_vmin, ec_vmax)
        elif vmin is not None or vmax is not None:
            edge_collection.set_clim(vmin, vmax)
            
        self.add_collection(edge_collection)
    else:
        collection.set_edgecolor(edge_colors)
    
    # Add collection to axes
    self.add_collection(collection)
    
    # Update axes limits
    self._update_limits(collection)
    
    return collection
