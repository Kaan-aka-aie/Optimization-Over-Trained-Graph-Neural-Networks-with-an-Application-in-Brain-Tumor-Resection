"""Compatibility names for historical instance pickles, with no graph generation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subnetworkParcels = json.loads((ROOT / 'config/subnetwork_parcels.json').read_text())

class TrainingInstance:
    """Historical pickle container. Instances are loaded, never regenerated."""
    pass
