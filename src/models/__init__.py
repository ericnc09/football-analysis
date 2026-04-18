"""Model zoo — canonical architectures for xG prediction.

Import paths (single source of truth):
    from src.models.hybrid_gcn import HybridXGModel
    from src.models.hybrid_gat import HybridGATModel

Do NOT redefine these classes inline in scripts or in `app.py`. A PyTorch
`load_state_dict` call matches parameters by name; inline re-definitions
that drift from this module silently break serving.
"""

from .hybrid_gat import HybridGATModel
from .hybrid_gcn import HybridXGModel

__all__ = ["HybridGATModel", "HybridXGModel"]
