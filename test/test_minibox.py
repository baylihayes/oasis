import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy
import pytest

from oasis import minibox

class TestGetMiniBoxId:
    """Test suite for minibox.get_mini_box_id function"""

    @pytest.mark.parametrize(
        "position, boxsize, minisize, expected, description",
        [
            # Basic single positions
            ([0.1, 0.1, 0.1], 10.0, 1.0, 0, "origin position"),
            ([1.5, 2.5, 3.5], 10.0, 1.0, 321, "interior position"),
            ([5.0, 5.0, 5.0], 10.0, 1.0, 555, "center position"),

            # Non-unit minisize
            ([3.0, 6.0, 9.0], 12.0, 2.0, 163, "non-unit minisize"),
            ([1.0, 1.0, 1.0], 3.0, 0.5, 86, "fractional minisize"),
            ([2.5, 2.5, 2.5], 5.0, 2.5, 7, "minisize equals half boxsize"),

            # Edge positions
            ([9.9, 9.9, 9.9], 10.0, 1.0, 999, "near upper boundary"),
            ([0.01, 0.01, 0.01], 10.0, 1.0, 0, "near lower boundary"),
        ],
    )
    def test_single_positions(self, position, boxsize, minisize, expected, description):
        """Test single position scenarios"""
        pos = numpy.array(position)
        result = minibox.get_mini_box_id(pos, boxsize, minisize)

        assert result == expected, f"Failed for {description}"
        assert isinstance(result, (int, numpy.integer)), f"Wrong return type for {description}"

    @pytest.mark.parametrize(
        "positions, boxsize, minisize, expected, description",
        [
            # Basic multiple positions
            ([[0.1, 0.1, 0.1], [1.5, 0.1, 0.1], [0.1, 1.5, 0.1], [0.1, 0.1, 1.5]],
             10.0, 1.0, [0, 1, 10, 100], "axis-aligned positions"),

            # Different grid cells
            ([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]],
             10.0, 1.0, [0, 111, 222], "diagonal positions"),

            # Mixed positions
            ([[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]],
             10.0, 1.0, [0, 999], "corner to corner"),
        ],
    )
    def test_multiple_positions(self, positions, boxsize, minisize, expected, description):
        """Test multiple position scenarios"""
        pos = numpy.array(positions)
        result = minibox.get_mini_box_id(pos, boxsize, minisize)
        expected_arr = numpy.array(expected)

        numpy.testing.assert_array_equal(result, expected_arr,
                                         err_msg=f"Failed for {description}")
        assert isinstance(
            result, numpy.ndarray), f"Wrong return type for {description}"
        assert result.shape == (
            len(positions),), f"Wrong shape for {description}"

    @pytest.mark.parametrize(
        "boundary_pos, boxsize, minisize, expected_grid, description",
        [
            # Upper boundary cases
            ([10.0, 5.0, 5.0], 10.0, 1.0, [9, 5, 5], "upper x boundary"),
            ([5.0, 10.0, 5.0], 10.0, 1.0, [5, 9, 5], "upper y boundary"),
            ([5.0, 5.0, 10.0], 10.0, 1.0, [5, 5, 9], "upper z boundary"),
            ([10.0, 10.0, 10.0], 10.0, 1.0, [9, 9, 9], "all upper boundaries"),

            # Lower boundary cases
            ([0.0, 0.5, 0.5], 10.0, 1.0, [0, 0, 0], "lower x boundary"),
            ([0.5, 0.0, 0.5], 10.0, 1.0, [0, 0, 0], "lower y boundary"),
            ([0.5, 0.5, 0.0], 10.0, 1.0, [0, 0, 0], "lower z boundary"),
            ([0.0, 0.0, 0.0], 10.0, 1.0, [0, 0, 0], "all lower boundaries"),

            # Different box sizes
            ([5.0, 5.0, 5.0], 5.0, 1.0, [4, 4, 4],
             "boundary with different boxsize"),
            ([2.0, 2.0, 2.0], 2.0, 0.5, [3, 3, 3],
             "boundary with fractional minisize"),
        ],
    )
    def test_boundary_conditions(self, boundary_pos, boxsize, minisize, expected_grid, description):
        """Test boundary condition handling"""
        pos = numpy.array([boundary_pos])
        result = minibox.get_mini_box_id(pos, boxsize, minisize)

        # Calculate expected ID from grid indices
        n_cells = int(numpy.ceil(boxsize / minisize))
        shift = numpy.array([1, n_cells, n_cells**2])
        expected_id = numpy.sum(shift * numpy.array(expected_grid))

        numpy.testing.assert_array_equal(result, [expected_id],
                                         err_msg=f"Failed for {description}")

    @pytest.mark.parametrize(
        "tolerance, description",
        [
            (1e-10, "very close to boundary"),
            (1e-12, "extremely close to boundary"),
            (1e-15, "machine precision boundary"),
        ],
    )
    def test_numerical_precision_boundaries(self, tolerance, description):
        """Test handling of positions very close to boundaries"""
        positions = numpy.array([
            [tolerance, 1.0, 1.0],           # Close to lower bound
            [10.0 - tolerance, 1.0, 1.0]    # Close to upper bound
        ])
        boxsize = 10.0
        minisize = 1.0

        result = minibox.get_mini_box_id(positions, boxsize, minisize)

        assert len(result) == 2, f"Wrong result length for {description}"
        assert all(id >= 0 for id in result), f"Negative IDs for {description}"
        assert isinstance(
            result, numpy.ndarray), f"Wrong type for {description}"

    @pytest.mark.parametrize(
        "error_type, position, boxsize, minisize, error_msg",
        [
            # Type errors
            (TypeError, [1.0, 1.0, 1.0], 10.0, 1.0,
             "must be a numpy array"),
            (TypeError, numpy.array([1.0, 1.0, 1.0]), "10.0", 1.0,
             "must be numeric"),
            (TypeError, numpy.array([1.0, 1.0, 1.0]), 10.0, "1.0",
             "must be numeric"),

            # Value errors - sizes
            (ValueError, numpy.array([1.0, 1.0, 1.0]), -5.0, 1.0,
             "boxsize must be zero or positive"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 10.0, -1.0,
             "minisize must be zero or positive"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 5.0, 10.0,
             "minisize cannot be larger than boxsize"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 0.0, 1.0,
             "boxsize must be non-zero"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 10.0, 0.0,
             "minisize must be non-zero"),

            # Value errors - shapes
            (ValueError, numpy.array([1.0, 1.0]), 10.0, 1.0,
             "Input position must have shape"),
            (ValueError, numpy.array([[1.0, 1.0], [2.0, 2.0]]), 10.0, 1.0,
             "Input position must have shape"),
            (ValueError, numpy.array([[[1.0, 1.0, 1.0]]]), 10.0, 1.0,
             "Input position must have shape"),

            # Value errors - bounds
            (ValueError, numpy.array([15.0, 5.0, 5.0]), 10.0, 1.0,
             "All coordinates must be within \\[0, 10.0\\]"),
            (ValueError, numpy.array([5.0, -1.0, 5.0]), 10.0, 1.0,
             "All coordinates must be within \\[0, 10.0\\]"),
            (ValueError, numpy.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]]), 10.0, 1.0,
             "All coordinates must be within \\[0, 10.0\\]"),
        ],
    )
    def test_input_validation(self, error_type, position, boxsize, minisize, error_msg):
        """Test comprehensive input validation"""
        with pytest.raises(error_type, match=error_msg):
            minibox.get_mini_box_id(position, boxsize, minisize)

    @pytest.mark.parametrize(
        "boxsize, minisize, test_fraction",
        [
            (1.0, 0.1, 0.5),      # Small box, fine grid
            (10.0, 1.0, 0.3),     # Medium box, unit grid
            (5.5, 0.7, 0.8),      # Non-integer ratios
            (100.0, 3.3, 0.1),    # Large box, coarse grid
            (2.5, 2.5, 0.9),      # Single cell case
            (7.0, 0.33, 0.7),     # Many cells case
        ],
    )
    def test_various_configurations(self, boxsize, minisize, test_fraction):
        """Test various box and minisize configurations"""
        # Test position at a fraction of the box size
        position = numpy.array([boxsize * test_fraction] * 3)

        result = minibox.get_mini_box_id(position, boxsize, minisize)

        assert isinstance(
            result, int), f"Wrong type for boxsize={boxsize}, minisize={minisize}"
        assert result >= 0, f"Negative ID for boxsize={boxsize}, minisize={minisize}"

        # Verify the result is within expected bounds
        n_cells = int(numpy.ceil(boxsize / minisize))
        max_possible_id = n_cells**3 - 1
        assert result <= max_possible_id, f"ID too large for boxsize={boxsize}, minisize={minisize}"

    def test_unique_ids_comprehensive(self):
        """Test that positions in different mini-boxes get unique IDs"""
        boxsize = 6.0
        minisize = 1.0

        # Generate positions in different mini-boxes
        positions = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Place position in center of each mini-box
                    pos = [i + 0.5, j + 0.5, k + 0.5]
                    positions.append(pos)

        positions = numpy.array(positions)
        result = minibox.get_mini_box_id(positions, boxsize, minisize)

        # All IDs should be unique
        unique_ids = numpy.unique(result)
        assert len(unique_ids) == len(result), "Non-unique IDs found"
        assert len(unique_ids) == 27, "Wrong number of unique IDs"  # 3^3 = 27

    def test_empty_and_special_arrays(self):
        """Test edge cases with empty and special arrays"""
        boxsize = 10.0
        minisize = 1.0

        # Empty array. Should raise ValueError
        empty_positions = numpy.empty((0, 3))
        with pytest.raises(ValueError, match="Input position must contain"):
            minibox.get_mini_box_id(empty_positions, boxsize, minisize)

        # Single row array
        single_pos = numpy.array([[5.0, 5.0, 5.0]])
        result = minibox.get_mini_box_id(single_pos, boxsize, minisize)
        assert result.size == 1
        assert isinstance(result, numpy.ndarray)

    def test_in_place_modification_behavior(self):
        """Test that the function modifies input arrays in-place"""
        # Test with boundary positions that will be modified
        original_positions = numpy.array([
            [10.0, 5.0, 5.0],   # At upper boundary
            [0.0, 5.0, 5.0],    # At lower boundary
            [5.0, 10.0, 5.0]    # At upper boundary
        ])

        # Keep a copy to compare
        positions_copy = original_positions.copy()
        boxsize = 10.0
        minisize = 1.0

        result = minibox.get_mini_box_id(original_positions, boxsize, minisize)

        # Check that modifications occurred
        modifications_found = not numpy.array_equal(
            original_positions, positions_copy)
        assert modifications_found, "Expected in-place modifications for boundary positions"

        # Check that boundary positions were adjusted inward
        assert numpy.all(original_positions <=
                         boxsize), "Positions not properly bounded"

        # Results should still be valid
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 3
        assert all(id >= 0 for id in result)

    @pytest.mark.parametrize(
        "n_positions",
        [1, 10, 100, 1000],
    )
    def test_scalability(self, n_positions):
        """Test function performance with different array sizes"""
        boxsize = 10.0
        minisize = 1.0

        # Generate random positions within bounds
        numpy.random.seed(42)  # For reproducibility
        positions = numpy.random.uniform(0.1, 9.9, size=(n_positions, 3))

        result = minibox.get_mini_box_id(positions, boxsize, minisize)

        assert isinstance(
            result, numpy.ndarray if n_positions > 0 else numpy.ndarray)
        assert result.shape == (n_positions,)
        assert all(id >= 0 for id in result)


class TestGetAdjacentMiniBoxIds:
    """Comprehensive test suite for get_adjacent_mini_box_ids function."""

    # Test input validation
    @pytest.mark.parametrize(
        "mini_box_id, expected_error",
        [
            (3.14, TypeError),
            ("5", TypeError),
            ([1], TypeError),
            (None, TypeError),
        ],
    )
    def test_invalid_mini_box_id_type(self, mini_box_id, expected_error):
        """Test that invalid mini_box_id types raise TypeError."""
        with pytest.raises(expected_error):
            minibox.get_adjacent_mini_box_ids(mini_box_id, 1.0, 1.0)

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_error",
        [
            ("1.0", 1.0, TypeError),
            (1.0, "1.0", TypeError),
            (None, 1.0, TypeError),
            (1.0, None, TypeError),
            ([1.0], 1.0, TypeError),
            (1.0, [1.0], TypeError),
        ],
    )
    def test_invalid_boxsize_minisize_types(self, boxsize, minisize, expected_error):
        """Test that invalid boxsize/minisize types raise TypeError."""
        with pytest.raises(expected_error):
            minibox.get_adjacent_mini_box_ids(0, boxsize, minisize)

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_error",
        [
            (0.0, 1.0, ValueError),
            (-1.0, 1.0, ValueError),
            (1.0, 0.0, ValueError),
            (1.0, -1.0, ValueError),
            (1.0, 2.0, ValueError),  # minisize > boxsize
        ],
    )
    def test_invalid_size_values(self, boxsize, minisize, expected_error):
        """Test that invalid size values raise ValueError."""
        with pytest.raises(expected_error):
            minibox.get_adjacent_mini_box_ids(0, boxsize, minisize)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize",
        [
            (-1, 2.0, 1.0),
            (-5, 2.0, 1.0),
            (8, 2.0, 1.0),  # 2x2x2 grid has IDs 0-7, so 8 is invalid
            (1000, 2.0, 1.0),
        ],
    )
    def test_mini_box_id_out_of_range(self, mini_box_id, boxsize, minisize):
        """Test that mini_box_id outside valid range raises ValueError."""
        with pytest.raises(ValueError):
            minibox.get_adjacent_mini_box_ids(mini_box_id, boxsize, minisize)

    # Test return value properties
    def test_return_type_and_shape(self):
        """Test that function returns correct type and shape."""
        result = minibox.get_adjacent_mini_box_ids(0, 2.0, 1.0)

        assert isinstance(result, numpy.ndarray)
        assert result.shape == (27,)
        assert result.dtype == numpy.int32

    def test_first_element_is_input_id(self):
        """Test that first element of result is always the input mini_box_id."""
        test_cases = [
            (0, 2.0, 1.0),
            (7, 2.0, 1.0),
            (13, 3.0, 1.0),
            (26, 3.0, 1.0),
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)
            assert result[0] == mini_box_id

    def test_get_adjacent_mini_box_ids(self):
        """Test that the adjacent IDs are in the correct order and are these explicit values."""
        l_box = 100.
        l_mb = 20.

        adj_ids = minibox.get_adjacent_mini_box_ids(
            mini_box_id=0,
            boxsize=l_box,
            minisize=l_mb,
        )

        adj_ids_0 = [0, 1, 4, 5, 6, 9, 20, 21, 24, 25, 26, 29, 30, 31, 34, 45,
                     46, 49, 100, 101, 104, 105, 106, 109, 120, 121, 124]

        assert len(adj_ids) == 27
        assert all([i in adj_ids for i in adj_ids_0])

    def test_all_ids_in_valid_range(self):
        """Test that all returned IDs are within valid range."""
        test_cases = [
            (0, 2.0, 1.0),   # 2x2x2 grid (8 boxes)
            (13, 3.0, 1.0),  # 3x3x3 grid (27 boxes)
            (50, 4.0, 1.0),  # 4x4x4 grid (64 boxes)
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            cells_per_side = int(numpy.ceil(boxsize / minisize))
            max_valid_id = cells_per_side**3 - 1

            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)

            assert numpy.all(result >= 0)
            assert numpy.all(result <= max_valid_id)

    def test_no_duplicate_ids(self):
        """Test that all returned IDs are unique."""
        test_cases = [
            (0, 2.0, 1.0),
            (13, 3.0, 1.0),
            (31, 4.0, 1.0),
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)
            unique_ids = numpy.unique(result)
            if boxsize / minisize == 2:
                assert len(unique_ids) == 8, "Duplicate IDs found"
            else:
                assert len(unique_ids) == 27, "Duplicate IDs found"

    # Test specific grid configurations

    def test_1x1x1_grid(self):
        """Test edge case of 1x1x1 grid - all neighbors should be the same box."""
        result = minibox.get_adjacent_mini_box_ids(0, 1.0, 1.0)

        # In a 1x1x1 grid, all 27 neighbors should be box 0 (itself)
        expected = numpy.full(27, 0, dtype=numpy.int32)
        numpy.testing.assert_array_equal(result, expected)

    def test_2x2x2_grid_all_positions(self):
        """Test all positions in a 2x2x2 grid."""
        boxsize, minisize = 2.0, 1.0

        # In a 2x2x2 grid, every box is adjacent to every other box due to wrapping
        for mini_box_id in range(8):
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)

            # Should contain all 8 box IDs, with many repetitions to make 27 total
            unique_ids = numpy.unique(result)
            expected_unique = numpy.arange(8, dtype=numpy.int32)
            numpy.testing.assert_array_equal(
                numpy.sort(unique_ids), expected_unique)

    def test_3x3x3_grid_center_box(self):
        """Test center box in 3x3x3 grid (ID 13)."""
        result = minibox.get_adjacent_mini_box_ids(13, 3.0, 1.0)

        # Center box (1,1,1) should have all 27 unique neighbors
        unique_ids = numpy.unique(result)
        assert len(unique_ids) == 27  # All boxes in 3x3x3 grid
        numpy.testing.assert_array_equal(
            numpy.sort(unique_ids), numpy.arange(27))

    def test_3x3x3_grid_corner_box(self):
        """Test corner box in 3x3x3 grid (ID 0)."""
        result = minibox.get_adjacent_mini_box_ids(0, 3.0, 1.0)

        # Corner box should still have 27 neighbors due to periodic boundaries
        assert len(result) == 27
        assert result[0] == 0  # First element is input ID

        # Verify periodic wrapping - corner should connect to opposite edges
        # Box 0 is at coordinates (0,0,0)
        # Its neighbors include boxes at coordinates like (2,2,2) due to wrapping
        expected_neighbors = {0, 1, 2, 3, 6, 9, 18, 19,
                              20, 21, 24, 26}  # Sample expected neighbors
        result_set = set(result)
        assert expected_neighbors.issubset(result_set)

    @pytest.mark.parametrize(
        "mini_box_id, expected_coords",
        [
            (0, (0, 0, 0)),
            (1, (0, 0, 1)),
            (3, (0, 1, 0)),
            (9, (1, 0, 0)),
            (13, (1, 1, 1)),  # center
            (26, (2, 2, 2)),  # far corner
        ],
    )
    def test_coordinate_conversion_3x3x3(self, mini_box_id, expected_coords):
        """Test that coordinate conversion is working correctly."""
        # This is an indirect test - we'll verify by checking that the function
        # doesn't raise RuntimeError (which would indicate coordinate conversion failure)
        result = minibox.get_adjacent_mini_box_ids(mini_box_id, 3.0, 1.0)
        assert len(result) == 27
        assert result[0] == mini_box_id

    # Test with different box/mini-box size ratios
    @pytest.mark.parametrize(
        "boxsize, minisize, expected_cells",
        [
            (2.0, 1.0, 2),    # Exact division
            (2.1, 1.0, 3),    # Ceil rounds up
            (3.9, 2.0, 2),    # Ceil rounds up
            (10.0, 3.0, 4),   # 10/3 = 3.33... -> 4 cells
        ],
    )
    def test_different_size_ratios(self, boxsize, minisize, expected_cells):
        """Test grid generation with different box/mini-box size ratios."""
        total_boxes = expected_cells**3
        max_valid_id = total_boxes - 1

        # Test with first and last valid IDs
        for test_id in [0, max_valid_id]:
            result = minibox.get_adjacent_mini_box_ids(
                test_id, boxsize, minisize)
            assert len(result) == 27
            assert result[0] == test_id
            assert numpy.all(result >= 0)
            assert numpy.all(result <= max_valid_id)

    # Test periodic boundary conditions explicitly
    def test_periodic_boundaries_4x4x4_edge_cases(self):
        """Test periodic boundary conditions with specific edge cases in 4x4x4 grid."""
        boxsize, minisize = 4.0, 1.0

        # Test box at edge (ID 3, coordinates (0,0,3))
        result = minibox.get_adjacent_mini_box_ids(3, boxsize, minisize)

        # Box at (0,0,3) should have neighbor at (0,0,0) due to k-direction wrapping
        assert 0 in result  # Periodic boundary neighbor
        assert 2 in result  # Regular neighbor
        assert 3 in result  # Self

        # Test box at face (ID 15, coordinates (0,3,3))
        result = minibox.get_adjacent_mini_box_ids(15, boxsize, minisize)

        # Should wrap in both j and k directions
        assert 12 in result  # k-direction wrap
        assert 3 in result   # j-direction wrap
        assert 0 in result   # Both j,k wrap

    def test_large_grid_performance(self):
        """Test function works with reasonably large grids."""
        # 10x10x10 grid (1000 boxes)
        result = minibox.get_adjacent_mini_box_ids(500, 10.0, 1.0)

        assert len(result) == 27
        assert result[0] == 500
        assert numpy.all(result >= 0)
        assert numpy.all(result <= 999)

    # Test floating-point edge cases
    def test_floating_point_precision(self):
        """Test with floating-point values that might cause precision issues."""
        test_cases = [
            (0, 2.000000001, 1.0),
            (0, 3.999999999, 2.0),
            (0, 1.0000000001, 0.5),
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)
            assert len(result) == 27
            assert result[0] == mini_box_id

    # Integration test with coordinate reconstruction
    def test_coordinate_reconstruction_consistency(self):
        """Test that coordinate conversion is consistent and reversible."""
        boxsize, minisize = 5.0, 1.0
        cells_per_side = 5

        # Test several random boxes
        test_ids = [0, 1, 5, 25, 62, 124]  # Various positions in 5x5x5 grid

        for test_id in test_ids:
            result = minibox.get_adjacent_mini_box_ids(
                test_id, boxsize, minisize)

            # Verify all neighbors are valid and unique
            assert len(result) == 27
            # Could be less due to wrapping
            assert len(numpy.unique(result)) <= 27
            assert numpy.all(result >= 0)
            assert numpy.all(result < cells_per_side**3)

            # Verify the input ID is preserved
            assert result[0] == test_id


class TestSplitSimulationIntoMiniBoxes:
    """Test suite for splitting simulation data into mini-box files.

    Input validation moved from split_simulation_into_mini_boxes() to the
    public entry point process_simulation_data(), which validates and then
    calls the splitter. These tests therefore go through
    process_simulation_data(). Files store each dataset under a particle-type
    group (e.g. 'dm/ID'), together with the particle mass.
    """

    PARTICLE_TYPE = 'dm'

    @staticmethod
    def split(save_path, positions, velocities, uid, boxsize, minisize,
              data=None, particle_type='dm', mass=(1.0, 'mass')):
        """Write mini-box files through the public entry point."""
        minibox.process_simulation_data(
            save_path=save_path,
            particle_type=particle_type,
            boxsize=boxsize,
            minisize=minisize,
            positions=positions,
            velocities=velocities,
            ids=uid,
            mass=mass,
            data=data,
            n_threads=1,
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def basic_simulation_data(self):
        """Generate basic simulation data for testing."""
        n_particles = 100
        positions = numpy.random.rand(n_particles, 3) * 10.0
        velocities = numpy.random.rand(n_particles, 3) * 5.0
        uid = numpy.arange(n_particles, dtype=numpy.uint32)
        return positions, velocities, uid

    @pytest.fixture
    def large_simulation_data(self):
        """Generate larger simulation data for performance testing."""
        n_particles = 10000
        positions = numpy.random.rand(n_particles, 3) * 100.0
        velocities = numpy.random.randn(n_particles, 3) * 10.0
        uid = numpy.arange(n_particles, dtype=numpy.uint64)
        return positions, velocities, uid

    @pytest.fixture
    def simulation_with_props(self):
        """Generate simulation data with additional properties."""
        n_particles = 200
        positions = numpy.random.rand(n_particles, 3) * 20.0
        velocities = numpy.random.rand(n_particles, 3) * 8.0
        uid = numpy.arange(n_particles, dtype=numpy.uint32)

        # Additional properties ('mass' is reserved for the particle mass)
        potentials = numpy.random.exponential(
            1e10, n_particles).astype(numpy.float32)
        temperatures = numpy.random.exponential(
            1e6, n_particles).astype(numpy.float32)

        props = (
            [potentials, temperatures],
            ['potential', 'temperature'],
            [numpy.float32, numpy.float32]
        )
        return positions, velocities, uid, props

    def verify_hdf5_structure(self, file_path: Path, expected_datasets: List[str],
                              name: Optional[str] = None):
        """Verify HDF5 file structure and datasets."""
        with h5py.File(file_path, 'r') as f:
            prefix = f"{name}/" if name else ""
            for dataset in expected_datasets:
                dataset_path = f"{prefix}{dataset}"
                assert dataset_path in f, f"Dataset {dataset_path} not found in {file_path}"
                assert isinstance(
                    f[dataset_path], h5py.Dataset), f"{dataset_path} is not a dataset"

    def verify_data_consistency(self, file_path: Path, original_positions: numpy.ndarray,
                                original_velocities: numpy.ndarray, original_uid: numpy.ndarray,
                                boxsize: float, minisize: float, mini_box_id: int,
                                name: Optional[str] = None, props: Optional[Tuple] = None):
        """Verify that saved data is consistent with original data."""
        with h5py.File(file_path, 'r') as f:
            prefix = f"{name}/" if name else ""

            # Load saved data
            saved_positions = f[f"{prefix}pos"][:]
            saved_velocities = f[f"{prefix}vel"][:]
            saved_uid = f[f"{prefix}ID"][:]

            # Verify shapes
            n_particles_in_box = saved_positions.shape[0]
            assert saved_velocities.shape[0] == n_particles_in_box
            assert saved_uid.shape[0] == n_particles_in_box

            if n_particles_in_box == 0:
                return  # No particles to verify further

            # Verify all particles belong to this mini-box
            computed_ids = minibox.get_mini_box_id(
                saved_positions, boxsize, minisize)
            if isinstance(computed_ids, numpy.ndarray):
                assert numpy.all(computed_ids == mini_box_id), \
                    f"Not all particles belong to mini-box {mini_box_id}"
            else:
                assert computed_ids == mini_box_id, \
                    f"Particle doesn't belong to mini-box {mini_box_id}"

            # Verify data integrity by checking if saved UIDs exist in original data
            original_uid_set = set(original_uid)
            for saved_id in saved_uid:
                assert saved_id in original_uid_set, \
                    f"Saved UID {saved_id} not found in original data"

            # If props are provided, verify them too
            if props:
                _, labels, dtypes = props
                for i, label in enumerate(labels):
                    saved_prop = f[f"{prefix}{label}"][:]
                    assert saved_prop.shape[0] == n_particles_in_box, \
                        f"Property {label} has inconsistent number of particles"
                    assert saved_prop.dtype == dtypes[i], \
                        f"Property {label} has incorrect dtype"

    def test_basic_functionality(self, temp_dir, basic_simulation_data):
        """Test basic functionality with simple data."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

        # Verify directory structure
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"
        assert output_dir.exists(), "Output directory not created"

        # Verify files were created
        hdf5_files = list(output_dir.glob("*.hdf5"))
        assert len(hdf5_files) > 0, "No HDF5 files created"

        # Verify file structure
        expected_datasets = ['ID', 'pos', 'vel', 'mass']
        for file_path in hdf5_files:
            self.verify_hdf5_structure(file_path, expected_datasets,
                                       self.PARTICLE_TYPE)

    def test_particle_type_group(self, temp_dir, basic_simulation_data):
        """Datasets are stored under the particle-type group, and a
        per-particle mass array is stored row by row (replaces the old
        'name' parameter, which no longer exists)."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0
        masses = numpy.arange(len(uid), dtype=numpy.float64) + 1.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize,
                   particle_type='gas', mass=(masses, 'mass'))

        # Verify directory structure
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        # Verify file structure with the particle-type prefix
        expected_datasets = ['ID', 'pos', 'vel', 'mass']
        hdf5_files = list(output_dir.glob("*.hdf5"))
        assert len(hdf5_files) > 0, "No HDF5 files created"
        for file_path in hdf5_files:
            self.verify_hdf5_structure(file_path, expected_datasets, 'gas')
            with h5py.File(file_path, 'r') as f:
                saved_ids = f['gas/ID'][:].astype(numpy.int64)
                numpy.testing.assert_allclose(f['gas/mass'][:], masses[saved_ids])

    def test_with_additional_properties(self, temp_dir, simulation_with_props):
        """Test functionality with additional particle properties."""
        positions, velocities, uid, props = simulation_with_props
        boxsize = 20.0
        minisize = 4.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize,
                   data=props)

        # Verify directory structure
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        # Verify file structure with additional properties
        expected_datasets = ['ID', 'pos', 'vel', 'mass', 'potential', 'temperature']
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            self.verify_hdf5_structure(file_path, expected_datasets,
                                       self.PARTICLE_TYPE)
            mini_box_id = int(file_path.stem)
            self.verify_data_consistency(
                file_path, positions, velocities, uid, boxsize, minisize,
                mini_box_id, self.PARTICLE_TYPE, props
            )

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_cells", 
        [
            (10.0, 2.0, 5),
            (10.0, 3.0, 4),
            (15.0, 5.0, 3),
            (20.0, 7.0, 3),
            (100.0, 25.0, 4)
        ],
    )
    def test_different_box_configurations(self, temp_dir, basic_simulation_data,
                                          boxsize, minisize, expected_cells):
        """Test with different box size configurations."""
        positions, velocities, uid = basic_simulation_data
        # Scale positions to fit the box
        positions = positions * (boxsize / 10.0)

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

        # Verify correct number of cells
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{expected_cells}"
        assert output_dir.exists(
        ), f"Expected directory with {expected_cells} cells per side"

    def test_data_consistency(self, temp_dir, basic_simulation_data):
        """Test that saved data is consistent with original data."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

        # Verify data consistency
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            mini_box_id = int(file_path.stem)
            self.verify_data_consistency(
                file_path, positions, velocities, uid,
                boxsize, minisize, mini_box_id, self.PARTICLE_TYPE
            )

    def test_particle_count_conservation(self, temp_dir, basic_simulation_data):
        """Test that all particles are saved and none are lost or duplicated."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

        # Count total particles across all files
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        total_saved_particles = 0
        all_saved_uids = []

        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                n_particles = f[f'{self.PARTICLE_TYPE}/ID'].shape[0]
                total_saved_particles += n_particles
                all_saved_uids.extend(f[f'{self.PARTICLE_TYPE}/ID'][:])

        # Verify particle count
        assert total_saved_particles == len(positions), \
            f"Particle count mismatch: original {len(positions)}, saved {total_saved_particles}"

        # Verify no duplicates
        assert len(set(all_saved_uids)) == len(all_saved_uids), \
            "Duplicate particles found in saved files"

        # Verify all original UIDs are present
        original_uid_set = set(uid)
        saved_uid_set = set(all_saved_uids)
        assert original_uid_set == saved_uid_set, \
            "Saved UIDs don't match original UIDs"

    def test_large_simulation(self, temp_dir, large_simulation_data):
        """Test with larger simulation data."""
        positions, velocities, uid = large_simulation_data
        boxsize = 100.0
        minisize = 10.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

        # Verify particle conservation
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        total_particles = 0
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                total_particles += f[f'{self.PARTICLE_TYPE}/ID'].shape[0]

        assert total_particles == len(positions), \
            f"Particle count mismatch in large simulation: {total_particles} vs {len(positions)}"

    @pytest.mark.parametrize("invalid_positions", [
        numpy.random.rand(100, 2),  # Wrong number of columns
        numpy.random.rand(100),     # 1D array
        numpy.random.rand(100, 3, 2),  # 3D array
    ])
    def test_invalid_positions_shape(self, temp_dir, invalid_positions):
        """Test validation of positions array shape."""
        velocities = numpy.random.rand(100, 3)
        uid = numpy.arange(100)

        with pytest.raises(ValueError, match="positions must have shape"):
            self.split(temp_dir + "/", invalid_positions, velocities, uid, 10.0, 2.0)

    @pytest.mark.parametrize("invalid_velocities", [
        numpy.random.rand(100, 2),  # Wrong number of columns
        numpy.random.rand(100),     # 1D array
        numpy.random.rand(50, 3),   # Wrong number of rows
    ])
    def test_invalid_velocities_shape(self, temp_dir, invalid_velocities):
        """Test validation of velocities array shape."""
        positions = numpy.random.rand(100, 3)
        uid = numpy.arange(100)

        with pytest.raises(ValueError):
            self.split(temp_dir + "/", positions, invalid_velocities, uid, 10.0, 2.0)

    @pytest.mark.parametrize("invalid_uid", [
        numpy.random.rand(100, 2),  # 2D array
        numpy.arange(50),           # Wrong length
    ])
    def test_invalid_uid_shape(self, temp_dir, invalid_uid):
        """Test validation of uid array shape."""
        positions = numpy.random.rand(100, 3)
        velocities = numpy.random.rand(100, 3)

        with pytest.raises(ValueError):
            self.split(temp_dir + "/", positions, velocities, invalid_uid, 10.0, 2.0)

    @pytest.mark.parametrize("boxsize,minisize,expected_error", [
        (0, 2.0, "boxsize must be non-zero"),
        (-5.0, 2.0, "boxsize must be zero or positive"),
        (10.0, 0, "minisize must be non-zero"),
        (10.0, -2.0, "minisize must be zero or positive"),
        (5.0, 10.0, "minisize cannot be larger than boxsize"),
        ("10.0", 2.0, "must be numeric"),
        (10.0, "2.0", "must be numeric"),
    ])
    def test_invalid_box_parameters(self, temp_dir, basic_simulation_data,
                                    boxsize, minisize, expected_error):
        """Test validation of box size parameters."""
        positions, velocities, uid = basic_simulation_data

        with pytest.raises((ValueError, TypeError), match=expected_error):
            self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

    def test_empty_arrays(self, temp_dir):
        """Test behavior with empty input arrays."""
        positions = numpy.empty((0, 3))
        velocities = numpy.empty((0, 3))
        uid = numpy.empty(0, dtype=int)

        with pytest.raises(ValueError, match="Input positions must contain at least one particle"):
            self.split(temp_dir + "/", positions, velocities, uid, 10.0, 2.0)

    def test_invalid_props_structure(self, temp_dir, basic_simulation_data):
        """Test validation of the extra-properties ('data') structure."""
        positions, velocities, uid = basic_simulation_data

        # Test with wrong tuple length
        invalid_props = ([numpy.random.rand(100)], ["mass"])  # Missing dtypes

        with pytest.raises(ValueError, match="data must be a tuple of"):
            self.split(temp_dir + "/", positions, velocities, uid, 10.0, 2.0,
                       data=invalid_props)

    def test_mismatched_props_arrays(self, temp_dir, basic_simulation_data):
        """Test validation of props arrays with mismatched lengths."""
        positions, velocities, uid = basic_simulation_data

        # Props array with wrong length
        invalid_props = (
            [numpy.random.rand(50)],  # Wrong length
            ["temperature"],
            [numpy.float32]
        )

        with pytest.raises(ValueError, match="must have.*elements"):
            self.split(temp_dir + "/", positions, velocities, uid, 10.0, 2.0,
                       data=invalid_props)

    def test_boundary_particles(self, temp_dir):
        """Test handling of particles at box boundaries."""
        # Create particles exactly at boundaries
        positions = numpy.array([
            [0.0, 0.0, 0.0],      # At origin
            [10.0, 10.0, 10.0],   # At upper boundary
            [5.0, 5.0, 5.0],      # In middle
            [9.999999, 9.999999, 9.999999],  # Very close to boundary
        ])
        velocities = numpy.random.rand(4, 3)
        uid = numpy.arange(4)

        boxsize = 10.0
        minisize = 5.0

        self.split(temp_dir + "/", positions, velocities, uid, boxsize, minisize)

        # Verify all particles are saved
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        total_particles = 0
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                total_particles += f[f'{self.PARTICLE_TYPE}/ID'].shape[0]

        assert total_particles == 4, "Not all boundary particles were saved"


# --------------------
# Low-level file makers
# --------------------
def _make_particle_file(path: Path, positions, velocities, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("part")
        grp.create_dataset("pos", data=positions)
        grp.create_dataset("vel", data=velocities)
        grp.create_dataset("ID", data=ids)
        grp.create_dataset("mass", data=1.0)     # constant particle mass


def _make_seed_file(path: Path, positions, velocities, ids, r200, m200, rs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("seed")
        grp.create_dataset("pos", data=positions)
        grp.create_dataset("vel", data=velocities)
        grp.create_dataset("ID", data=ids)
        grp.create_dataset("R200b", data=r200)
        grp.create_dataset("M200b", data=m200)
        grp.create_dataset("Rs", data=rs)


# ------------------------
# High-level test utilities
# ------------------------
def make_all_particle_files(tmp_path, mini_box_id, boxsize, minisize, with_particle=True):
    """Create particle files for target + neighbors. Only target has one particle if with_particle=True."""
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    mini_box_ids = minibox.get_adjacent_mini_box_ids(
        mini_box_id, boxsize, minisize)
    subdir = tmp_path / f"mini_boxes_nside_{cells_per_side}"

    # Compute center of target mini-box
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side
    center = numpy.array([(k + 0.5) * minisize, (j + 0.5)
                         * minisize, (i + 0.5) * minisize])

    for mb in mini_box_ids:
        file_path = subdir / f"{mb}.hdf5"
        if mb == mini_box_id and with_particle:
            _make_particle_file(
                file_path,
                positions=numpy.array([center]),
                velocities=numpy.array([[1.0, 1.0, 1.0]]),
                ids=numpy.array([42], dtype=int),
            )
        else:
            _make_particle_file(
                file_path,
                positions=numpy.empty((0, 3)),
                velocities=numpy.empty((0, 3)),
                ids=numpy.empty((0,), dtype=int),
            )


def make_all_seed_files(tmp_path, mini_box_id, boxsize, minisize, with_seed=True):
    """Create seed files for target + neighbors. Only target has one seed if with_seed=True."""
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    mini_box_ids = minibox.get_adjacent_mini_box_ids(
        mini_box_id, boxsize, minisize)
    subdir = tmp_path / f"mini_boxes_nside_{cells_per_side}"

    # Compute center of target mini-box
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side
    center = numpy.array([(k + 0.5) * minisize, (j + 0.5)
                         * minisize, (i + 0.5) * minisize])

    for mb in mini_box_ids:
        file_path = subdir / f"{mb}.hdf5"
        if mb == mini_box_id and with_seed:
            _make_seed_file(
                file_path,
                positions=numpy.array([center]),
                velocities=numpy.array([[1.0, 1.0, 1.0]]),
                ids=numpy.array([7], dtype=int),
                r200=numpy.array([10.0]),
                m200=numpy.array([20.0]),
                rs=numpy.array([1.0]),
            )
        else:
            _make_seed_file(
                file_path,
                positions=numpy.empty((0, 3)),
                velocities=numpy.empty((0, 3)),
                ids=numpy.empty((0,), dtype=int),
                r200=numpy.empty((0,)),
                m200=numpy.empty((0,)),
                rs=numpy.empty((0,)),
            )


class TestLoadParticles:
    def test_valid_inputs(self, tmp_path):
        make_all_particle_files(tmp_path, mini_box_id=0,
                                boxsize=10.0, minisize=1.0, with_particle=True)

        pos, vel, ids, mass = minibox.load_particles(
            0, 10.0, 1.0, str(tmp_path) + "/", particle_type="part", padding=1.0
        )

        assert pos.shape == (1, 3)
        assert vel.shape == (1, 3)
        assert ids.shape == (1,)
        assert int(ids[0]) == 42
        assert float(mass) == 1.0

    def test_runtime_error_when_no_particles(self, tmp_path):
        make_all_particle_files(tmp_path, 0, 10.0, 10.0, with_particle=False)

        with pytest.raises(ValueError, match="must contain"):
            minibox.load_particles(
                0, 10.0, 10.0, str(tmp_path) + "/", particle_type="part",
                padding=1.0)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, padding, expected_error",
        [
            ("0", 10.0, 10.0, 1.0, TypeError),
            (0, "10", 10.0, 1.0, TypeError),
            (0, 10.0, "10", 1.0, TypeError),
            (0, 10.0, 10.0, "1.0", TypeError),
        ],
    )
    def test_type_errors(self, tmp_path, mini_box_id, boxsize, minisize, padding, expected_error):
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()
        
        with pytest.raises(expected_error):
            minibox.load_particles(mini_box_id, boxsize, minisize, valid_dir,
                                   particle_type="part", padding=padding)

    def test_value_errors_and_file_errors(self, tmp_path):
        subdir = tmp_path / "mini_boxes_nside_1"
        subdir.mkdir()

        # Negative ID
        with pytest.raises(ValueError, match="mini_box_id must be zero or positive"):
            minibox.load_particles(-1, 10.0, 10.0, str(tmp_path) + "/",
                                   particle_type="part", padding=1.0)

        # Not a directory
        file_path = tmp_path / "not_a_dir.hdf5"
        file_path.write_text("x")
        with pytest.raises(NotADirectoryError):
            minibox.load_particles(0, 10.0, 10.0, file_path,
                                   particle_type="part", padding=1.0)

    def test_missing_mini_box_folder_raises(self, tmp_path):
        """A missing mini_boxes_nside_<n>/ folder means a wrong load_path or
        minisize, not an empty region."""
        with pytest.raises(FileNotFoundError, match="Minibox folder not found"):
            minibox.load_particles(0, 10.0, 10.0, str(tmp_path) + "/",
                                   particle_type="part", padding=1.0)

    def test_wrong_particle_type_raises(self, tmp_path):
        """A particle type found in none of the files is reported by name."""
        make_all_particle_files(tmp_path, mini_box_id=0,
                                boxsize=10.0, minisize=1.0, with_particle=True)

        with pytest.raises(ValueError, match="No 'dm' particles found"):
            minibox.load_particles(0, 10.0, 1.0, str(tmp_path) + "/",
                                   particle_type="dm", padding=1.0)

    def test_empty_neighbours_do_not_drop_the_rest(self, tmp_path):
        """A neighbour without a file (empty mini-box) or with seeds only is
        skipped, and every other neighbour is still read. The original code
        stopped reading at the first such neighbour and silently dropped the
        particles of all the following ones."""
        boxsize, minisize, padding = 30.0, 10.0, 1.0
        target = 13                                   # centre of a 3x3x3 grid
        center = numpy.array([15.0, 15.0, 15.0])
        neighbours = list(dict.fromkeys(
            minibox.get_adjacent_mini_box_ids(target, boxsize, minisize).tolist()))
        missing, seeds_only = neighbours[1], neighbours[2]

        # One particle in each of the 27 mini-boxes, 5.5 from the target's
        # centre along each offset axis: inside the neighbour, and within
        # minisize / 2 + padding = 6 of the target's centre.
        offsets = numpy.array([[i, j, k] for i in (-1, 0, 1)
                               for j in (-1, 0, 1) for k in (-1, 0, 1)])
        positions = center + 5.5 * offsets
        box_of = minibox.get_mini_box_id(positions, boxsize, minisize)
        subdir = tmp_path / "mini_boxes_nside_3"
        for box in neighbours:
            rows = numpy.flatnonzero(box_of == box)
            if box == missing:
                continue                              # no file at all
            if box == seeds_only:
                _make_seed_file(subdir / f"{box}.hdf5", positions[rows],
                                numpy.zeros((len(rows), 3)), rows,
                                numpy.ones(len(rows)), numpy.ones(len(rows)),
                                numpy.ones(len(rows)))
                continue
            _make_particle_file(subdir / f"{box}.hdf5", positions[rows],
                                numpy.zeros((len(rows), 3)), rows)

        _, _, ids, _ = minibox.load_particles(
            target, boxsize, minisize, str(tmp_path) + "/",
            particle_type="part", padding=padding)

        expected = numpy.flatnonzero((box_of != missing) & (box_of != seeds_only))
        assert sorted(ids.tolist()) == expected.tolist()

    def test_load_path_as_path_object(self, tmp_path):
        """load_path may be a pathlib.Path, as documented."""
        make_all_particle_files(tmp_path, mini_box_id=0,
                                boxsize=10.0, minisize=1.0, with_particle=True)

        _, _, ids, _ = minibox.load_particles(
            0, 10.0, 1.0, tmp_path, particle_type="part", padding=1.0)
        assert ids.tolist() == [42]


# --------------
# Tests: Seeds
# --------------
class TestLoadSeeds:
    def test_valid_inputs(self, tmp_path):
        make_all_seed_files(tmp_path, mini_box_id=0,
                            boxsize=10.0, minisize=1.0, with_seed=True)

        pos, vel, ids, r200, m200, rs, mask = minibox.load_seeds(
            0, 10.0, 1.0, str(tmp_path) + "/", padding=1.0
        )

        assert pos.shape == (1, 3)
        assert vel.shape == (1, 3)
        assert ids.shape == (1,)
        assert r200.shape == (1,)
        assert m200.shape == (1,)
        assert rs.shape == (1,)
        assert mask.shape == (1,)
        assert int(ids[0]) == 7
        assert pytest.approx(float(m200[0])) == 20.0

    def test_runtime_error_when_no_seeds(self, tmp_path):
        make_all_seed_files(tmp_path, 0, 10.0, 10.0, with_seed=False)

        with pytest.raises(ValueError, match="must contain"):
            minibox.load_seeds(0, 10.0, 10.0, str(tmp_path) + "/", padding=1.0)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, padding, expected_error",
        [
            ("0", 10.0, 10.0, 1.0, TypeError),
            (0, "10", 10.0, 1.0, TypeError),
            (0, 10.0, "10", 1.0, TypeError),
            (0, 10.0, 10.0, "1.0", TypeError),
        ],
    )
    def test_type_errors(self, tmp_path, mini_box_id, boxsize, minisize, padding, expected_error):
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()

        with pytest.raises(expected_error):
            minibox.load_seeds(mini_box_id, boxsize, minisize, valid_dir,
                               padding=padding)

    def test_value_errors_and_file_errors(self, tmp_path):
        subdir = tmp_path / "mini_boxes_nside_1"
        subdir.mkdir()

        # Negative ID
        with pytest.raises(ValueError, match="mini_box_id must be zero or positive"):
            minibox.load_seeds(-1, 10.0, 10.0, str(tmp_path) + "/", padding=1.0)

        # Missing file
        with pytest.raises(FileNotFoundError):
            minibox.load_seeds(0, 10.0, 10.0, str(tmp_path) + "/", padding=1.0)

        # Not a directory
        file_path = tmp_path / "not_a_dir.hdf5"
        file_path.write_text("x")
        with pytest.raises(NotADirectoryError):
            minibox.load_seeds(0, 10.0, 10.0, file_path, padding=1.0)
