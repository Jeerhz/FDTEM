"""The two metric families, the named training mixes and the training presets."""
from __future__ import annotations

from pydantic import BaseModel

from part2_length_training.models import DIR_PREFIX, MixSpec, TrainPreset

FRACS = (0, 10, 20, 40, 60, 80, 100)  # sentence fraction of the fracNNN ladder, in %
K_LONG = (2, 3, 4, 6)                 # window sizes of the aggregated pool


class Family(BaseModel):
    hub_id: str
    base_cfg: str      # relative to the part directory
    dir_prefix: str    # checkpoint directory prefix


FAMILIES = {
    "da": Family(hub_id="Unbabel/wmt22-comet-da", base_cfg="configs/comet_da.yaml",
                 dir_prefix=DIR_PREFIX["da"]),
    "qe": Family(hub_id="Unbabel/wmt22-cometkiwi-da", base_cfg="configs/comet_qe.yaml",
                 dir_prefix=DIR_PREFIX["qe"]),
}

# Arms are named by what they were trained on (French labels used by the figures).
_LADDER_LABEL = {0: "mixte (concat. + natifs)", 100: "phrases"}
MIX_SPECS: dict[str, MixSpec] = {
    f"frac{f:03d}": MixSpec(name=f"frac{f:03d}", kind="mix", sentence_fraction=f,
                            description=_LADDER_LABEL.get(f, f"{f} % phrases"))
    for f in FRACS
}
MIX_SPECS["frac000nat"] = MixSpec(name="frac000nat", kind="nat", sentence_fraction=0,
                                  description="documents natifs")
MIX_SPECS["frac000agg"] = MixSpec(name="frac000agg", kind="agg", sentence_fraction=0,
                                  description="phrases concaténées")
MIX_SPECS["uncontrolled"] = MixSpec(name="uncontrolled", kind="uncontrolled", sentence_fraction=0,
                                    description="non contrôlé (contaminé)")
BASE_LABEL = "publié"  # the released metric, evaluated as is

TRAIN_PRESETS = {
    "default": TrainPreset(max_epochs=60, patience=10),
    "wave1": TrainPreset(max_epochs=6, patience=3),
    # everything, unfrozen from step 0, 10x the continue-training rates, early stopping off
    "uncontrolled": TrainPreset(max_epochs=60, patience=1000, encoder_lr=5e-6, head_lr=1e-4,
                                nr_frozen_epochs=0, val_files_from_mix=True),
}
