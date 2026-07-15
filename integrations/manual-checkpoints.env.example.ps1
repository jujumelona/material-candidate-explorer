# Copy this file outside the repository, replace every placeholder, then dot-source it.
# Automatic Hugging Face snapshots (MatterGen, Uni-Mol, ESM, RNA-FM, UMA, Boltz)
# are resolved from InstallRoot by start-sidecars.ps1 and must not be overridden here.

# Chemprop: one task checkpoint plus the exact output order and scientific units.
$env:CHEMPROP_CHECKPOINT_PATH = "D:\models\chemprop\task.ckpt"
$env:CHEMPROP_WEIGHT_REVISION = "sha256:<64-lowercase-hex>"
$env:CHEMPROP_PROPERTY_NAMES = "solubility,toxicity"
$env:CHEMPROP_PROPERTY_UNITS = "mol/L,dimensionless"

# REINVENT4: choose the reviewed prior explicitly; the launcher verifies its bytes.
$env:REINVENT_MODEL_FILE = "D:\models\reinvent\prior.model"
$env:REINVENT_WEIGHT_REVISION = "sha256:<64-lowercase-hex>"
$env:REINVENT_MODE = "reinvent"

# scGPT: the directory must contain args.json, vocab.json, and best_model.pt.
$env:SCGPT_CHECKPOINT_DIR = "D:\models\scgpt\whole-human"
$env:SCGPT_MAX_LENGTH = "1200"
$env:SCGPT_USE_FAST_TRANSFORMER = "false"

# QHNet: the checkpoint and its strict dataset/config scope form one bundle.
$env:QHNET_CHECKPOINT_PATH = "D:\models\qhnet\water_results.pt"
$env:QHNET_CONFIG_PATH = "$PSScriptRoot\qhnet.water.config.json"

# Upstream-managed weights cannot honestly claim a byte hash until a local file is bound.
$env:MATTERSIM_WEIGHT_REVISION = "managed-unattested:<release-and-cache-identity>"
$env:CHGNET_WEIGHT_REVISION = "managed-unattested:chgnet-0.3.0-builtin"

# Gated automatic snapshot. Obtain approval first; do not save the real token in this file.
$env:HF_TOKEN = "<temporary-hugging-face-token>"

