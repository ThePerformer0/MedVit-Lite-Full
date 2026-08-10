#!/bin/bash
# ============================================================
# MedViT-Lite — Script d'entraînement complet sur CloudLab
# ============================================================
# Lance toutes les expériences dans l'ordre correct :
#   1. Baseline ResNet-50
#   2. MedViT-Lite complet
#   3. Ablations (w/o DPS, w/o SFC, w/o DPS+SFC)
#
# Usage :
#   conda activate medvit
#   bash scripts/run_experiments.sh
#   bash scripts/run_experiments.sh --baseline-only  (juste ResNet-50)
#   bash scripts/run_experiments.sh --medvit-only    (juste MedViT-Lite)
#
# Les résultats seront dans : ./results/
# Les checkpoints dans      : ./checkpoints/

set -e  # Arrêt immédiat si erreur

BASELINE_ONLY=false
MEDVIT_ONLY=false
NO_WANDB=""

for arg in "$@"; do
    case $arg in
        --baseline-only) BASELINE_ONLY=true ;;
        --medvit-only)   MEDVIT_ONLY=true ;;
        --no-wandb)      NO_WANDB="--no-wandb" ;;
    esac
done

CONFIG="configs/base.yaml"
LOG_DIR="logs"
mkdir -p $LOG_DIR results checkpoints

echo "=============================================="
echo "  MedViT-Lite — Lancement des expériences"
echo "  $(date)"
echo "=============================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 1. Baseline ResNet-50 ────────────────────────────────────
if [ "$MEDVIT_ONLY" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [1/5] Baseline : ResNet-50"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/baseline_cnn.py \
        --config $CONFIG $NO_WANDB \
        2>&1 | tee $LOG_DIR/baseline_resnet50.log
    echo "✅ ResNet-50 terminé"
fi

if [ "$BASELINE_ONLY" = true ]; then
    echo "Mode --baseline-only : fin."
    exit 0
fi

# ── 2. MedViT-Lite complet (toutes innovations) ──────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/5] MedViT-Lite (DPS + SFC + HTA)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python experiments/medvit_lite_train.py \
    --config $CONFIG $NO_WANDB \
    2>&1 | tee $LOG_DIR/medvit_lite_full.log
echo "✅ MedViT-Lite complet terminé"

# ── 3. Ablation : sans DPS ───────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/5] Ablation : MedViT-Lite sans DPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python experiments/medvit_lite_train.py \
    --config $CONFIG $NO_WANDB --no-dps \
    2>&1 | tee $LOG_DIR/medvit_lite_noDPS.log
echo "✅ Ablation sans DPS terminée"

# ── 4. Ablation : sans SFC ───────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/5] Ablation : MedViT-Lite sans SFC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python experiments/medvit_lite_train.py \
    --config $CONFIG $NO_WANDB --no-sfc \
    2>&1 | tee $LOG_DIR/medvit_lite_noSFC.log
echo "✅ Ablation sans SFC terminée"

# ── 5. Ablation : sans aucune innovation (= ViT de base) ─────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/5] Ablation : MedViT-Lite sans innovations (ViT baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python experiments/medvit_lite_train.py \
    --config $CONFIG $NO_WANDB --no-dps --no-sfc \
    2>&1 | tee $LOG_DIR/medvit_lite_noInnovations.log
echo "✅ Ablation sans innovations terminée"

# ── Résumé final ─────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Toutes les expériences terminées !"
echo "  $(date)"
echo ""
echo "  Résultats dans : ./results/"
echo "  Checkpoints    : ./checkpoints/"
echo "  Logs           : ./logs/"
echo ""
echo "  Prochaine étape :"
echo "  python experiments/compare_results.py"
echo "=============================================="
