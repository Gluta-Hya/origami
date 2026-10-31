"""origami_cone: tangent-cone decomposition of rigid origami at the flat state."""
from .patterns import (Pattern, from_vertices, single_vertex,
                       single_vertex_sectors, miura, raise_degree,
                       raise_degree_by_index, raise_degree_at, perturb)
from .cone import (first_order_matrix, hessian_blocks, kernel, classify_block,
                   signature, decompose, cell_directions, Cell,
                   linear_cells, definiteness_count,
                   predicted_components, predicted_at,
                   opposite_pair_values,
                   incremental_linear_cells, fast_component_count,
                   cell_budget, narrow_side, predict_pattern)
from .kinematics import (residual, jacobian, local_dof, continue_branch,
                         find_folded_states, solve_slice, certify_cells,
                         certified_branch_count)
from .invariants import mv_pattern, cell_mv_patterns, mv_report, forcing_sets
from .patterns import (flat_foldable_quad, second_order_impostor,
                       rigid_foldable_no_straight_lines)
from . import criterion
from .criterion import gram_determinant, family_dies, predict_N
from . import tachi
from . import h3

__all__ = [n for n in dir() if not n.startswith("_")]
__version__ = "2.4.0"
