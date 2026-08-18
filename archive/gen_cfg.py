import sys, glob, os
import yaml

base_cfg, out_cfg, mix_path, frozen = sys.argv[1:5]
cfg = yaml.safe_load(open(base_cfg))
base_dir = os.path.dirname(os.path.abspath(base_cfg))

# sub-config references are relative to the base config's directory — make them
# absolute so the generated file can live anywhere
for key in ("trainer", "early_stopping", "model_checkpoint"):
    v = cfg.get(key)
    if isinstance(v, str):
        cfg[key] = os.path.normpath(os.path.join(base_dir, v))

metric_key = next(k for k, v in cfg.items()
                  if isinstance(v, dict) and "class_path" in v)
init = cfg[metric_key]["init_args"]

if mix_path:
    files = sorted(f for f in glob.glob(os.path.join(mix_path, "*_train.csv"))
                   if os.path.basename(f) != "all_train.csv")
    if not files:
        sys.exit(f"ERROR: no per-LP *_train.csv in {mix_path}")
    init["train_data"] = files
    print(f"train_data → {len(files)} files from {mix_path}")

if frozen == "1":
    # comet unfreezes once epoch_nr >= nr_frozen_epochs — this never fires
    init["nr_frozen_epochs"] = 1000
    init["keep_embeddings_frozen"] = True
    print("encoder frozen for the whole run (nr_frozen_epochs=1000)")

os.makedirs(os.path.dirname(os.path.abspath(out_cfg)), exist_ok=True)
with open(out_cfg, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
print(f"generated config → {out_cfg}")
