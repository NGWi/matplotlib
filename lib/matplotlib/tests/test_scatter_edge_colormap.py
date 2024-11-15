"""Tests for edge color mapping in scatter plots."""

import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal


@check_figures_equal(extensions=["png"])
def test_scatter_edge_colormap_basic(fig_test, fig_ref):
    """Test basic edge color mapping functionality."""
    # Generate sample data
    np.random.seed(19680801)
    x = np.random.rand(50)
    y = np.random.rand(50)
    edge_values = np.random.rand(50)
    face_values = np.random.rand(50)
    
    # Reference plot - using single colors
    ax_ref = fig_ref.add_subplot(111)
    ax_ref.scatter(x, y, c='blue', edgecolors='red')
    
    # Test plot - using colormaps
    ax_test = fig_test.add_subplot(111)
    ax_test.scatter(x, y, c=face_values, ec=edge_values, 
                   cmap='viridis', ec_cmap='plasma')


@check_figures_equal(extensions=["png"])
def test_scatter_edge_colormap_separate_norms(fig_test, fig_ref):
    """Test separate normalizations for face and edge colors."""
    np.random.seed(19680801)
    x = np.random.rand(50)
    y = np.random.rand(50)
    edge_values = np.random.rand(50)
    face_values = np.random.rand(50)
    
    # Reference plot
    ax_ref = fig_ref.add_subplot(111)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ax_ref.scatter(x, y, c=face_values, edgecolors='red', 
                  norm=norm, cmap='viridis')
    
    # Test plot - separate norms
    ax_test = fig_test.add_subplot(111)
    face_norm = mpl.colors.Normalize(vmin=0, vmax=1)
    edge_norm = mpl.colors.Normalize(vmin=0.2, vmax=0.8)
    ax_test.scatter(x, y, c=face_values, ec=edge_values,
                   norm=face_norm, ec_norm=edge_norm,
                   cmap='viridis', ec_cmap='plasma')


@check_figures_equal(extensions=["png"])
def test_scatter_edge_colormap_precedence(fig_test, fig_ref):
    """Test color precedence with edge colormapping."""
    np.random.seed(19680801)
    x = np.random.rand(50)
    y = np.random.rand(50)
    edge_values = np.random.rand(50)
    
    # Reference plot
    ax_ref = fig_ref.add_subplot(111)
    ax_ref.scatter(x, y, c='blue', edgecolors='red')
    
    # Test plot - ec should take precedence over edgecolors
    ax_test = fig_test.add_subplot(111)
    ax_test.scatter(x, y, c='blue', ec=edge_values, 
                   edgecolors='red', ec_cmap='plasma')


def test_scatter_edge_colormap_validation():
    """Test validation of edge colormap parameters."""
    x = np.random.rand(50)
    y = np.random.rand(50)
    edge_values = np.random.rand(50)
    
    fig, ax = plt.subplots()
    
    # Test mismatched sizes
    with pytest.raises(ValueError):
        ax.scatter(x, y, ec=edge_values[:-1])
    
    # Test invalid edge_values type
    with pytest.raises(ValueError):
        ax.scatter(x, y, ec="invalid")
    
    plt.close(fig)


@check_figures_equal(extensions=["png"])
def test_scatter_edge_colormap_with_alpha(fig_test, fig_ref):
    """Test edge colormapping with alpha values."""
    np.random.seed(19680801)
    x = np.random.rand(50)
    y = np.random.rand(50)
    edge_values = np.random.rand(50)
    alphas = np.linspace(0.1, 1.0, 50)
    
    # Reference plot
    ax_ref = fig_ref.add_subplot(111)
    ax_ref.scatter(x, y, c='blue', edgecolors='red', alpha=0.5)
    
    # Test plot
    ax_test = fig_test.add_subplot(111)
    ax_test.scatter(x, y, c='blue', ec=edge_values, 
                   ec_cmap='plasma', alpha=alphas)


import numpy as np
import pytest

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PathCollection


def test_scatter_color_mapping_combinations():
    """Test various combinations of face and edge color mapping parameters."""
    # Setup
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    c = x  # Values for face colors
    ec = y  # Values for edge colors
    
    # Test case 1: Only face color mapping parameters
    scatter1 = ax.scatter(x, y, c=c, cmap='viridis', norm=mcolors.Normalize(0, 10),
                         vmin=0, vmax=10, ec=None)
    assert isinstance(scatter1, PathCollection)
    assert scatter1.get_cmap().name == 'viridis'
    assert scatter1.norm.vmin == 0
    assert scatter1.norm.vmax == 10
    
    # Test case 2: Only edge color mapping parameters
    scatter2 = ax.scatter(x, y, c=None, ec=ec, ec_map='plasma',
                         ec_norm=mcolors.Normalize(0, 10),
                         ec_vmin=0, ec_vmax=10)
    collections = ax.collections
    edge_coll = collections[-1]  # Last collection should be edge collection
    assert edge_coll.get_cmap().name == 'plasma'
    assert edge_coll.norm.vmin == 0
    assert edge_coll.norm.vmax == 10
    
    # Test case 3: Edge colors inherit face color parameters
    scatter3 = ax.scatter(x, y, c=c, cmap='viridis',
                         norm=mcolors.Normalize(0, 10),
                         vmin=0, vmax=10,
                         ec=ec)  # No ec_* parameters
    collections = ax.collections
    edge_coll = collections[-1]
    assert edge_coll.get_cmap().name == 'viridis'  # Should inherit from face
    assert edge_coll.norm.vmin == 0
    assert edge_coll.norm.vmax == 10
    
    # Test case 4: Mix of face and edge parameters
    scatter4 = ax.scatter(x, y, c=c, cmap='viridis',
                         norm=mcolors.Normalize(0, 10),
                         vmin=0, vmax=10,
                         ec=ec, ec_map='plasma')  # Only ec_map specified
    collections = ax.collections
    edge_coll = collections[-1]
    assert edge_coll.get_cmap().name == 'plasma'  # Should use specified ec_map
    assert edge_coll.norm.vmin == 0  # Should inherit from face
    assert edge_coll.norm.vmax == 10
    
    # Test case 5: Edge parameters override face parameters
    scatter5 = ax.scatter(x, y, c=c, cmap='viridis',
                         norm=mcolors.Normalize(0, 10),
                         vmin=0, vmax=10,
                         ec=ec, ec_map='plasma',
                         ec_norm=mcolors.Normalize(-5, 15),
                         ec_vmin=-5, ec_vmax=15)
    collections = ax.collections
    edge_coll = collections[-1]
    assert edge_coll.get_cmap().name == 'plasma'
    assert edge_coll.norm.vmin == -5
    assert edge_coll.norm.vmax == 15
    
    plt.close()


def test_scatter_color_validation():
    """Test validation of color mapping parameters."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    c = x
    ec = y
    
    # Test case 1: Invalid norm and vmin/vmax combination for face colors
    with pytest.raises(ValueError, match="Passing a Normalize instance simultaneously with vmin/vmax"):
        ax.scatter(x, y, c=c, norm=mcolors.Normalize(0, 10), vmin=0, vmax=10)
    
    # Test case 2: Invalid norm and vmin/vmax combination for edge colors
    with pytest.raises(ValueError, match="Passing a Normalize instance simultaneously with ec_vmin/ec_vmax"):
        ax.scatter(x, y, ec=ec, ec_norm=mcolors.Normalize(0, 10),
                  ec_vmin=0, ec_vmax=10)
    
    plt.close()


@image_comparison(['scatter_edge_colormap_basic.png'])
def test_scatter_edge_colormap_basic():
    """Test basic edge color mapping visualization."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    
    # Create scatter plot with different mappings for face and edge
    ax.scatter(x, y, c=x, cmap='viridis', vmin=0, vmax=10,
              ec=y, ec_map='plasma', ec_vmin=0, ec_vmax=10,
              s=200, linewidths=3)
    
    plt.colorbar(ax.collections[0], label='Face Color')
    plt.colorbar(ax.collections[1], label='Edge Color')


@image_comparison(['scatter_edge_colormap_inherit.png'])
def test_scatter_edge_colormap_inherit():
    """Test edge colors inheriting face color parameters."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    
    # Create scatter plot where edge colors inherit face color parameters
    ax.scatter(x, y, c=x, cmap='viridis', vmin=0, vmax=10,
              ec=y, s=200, linewidths=3)  # No ec_* parameters
    
    plt.colorbar(ax.collections[0], label='Face Color')
    plt.colorbar(ax.collections[1], label='Edge Color (Inherited)')


def test_scatter_edge_colormap_data_validation():
    """Test validation of color data."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    
    # Test case 1: Mismatched lengths
    with pytest.raises(ValueError, match="x and y must be the same size"):
        ax.scatter(x, y[:-1], ec=y)
    
    # Test case 2: Invalid color specifications
    with pytest.raises(ValueError):
        ax.scatter(x, y, ec=[1, 2])  # Wrong length for ec
    
    plt.close()


def test_scatter_edge_colormap_alpha():
    """Test alpha handling with edge colormaps."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    
    # Test that alpha works with edge colormaps
    scatter = ax.scatter(x, y, c=x, cmap='viridis',
                        ec=y, ec_map='plasma',
                        alpha=0.5)
    
    assert scatter.get_alpha() == 0.5
    collections = ax.collections
    edge_coll = collections[-1]
    assert edge_coll.get_alpha() == 0.5
    
    plt.close()
