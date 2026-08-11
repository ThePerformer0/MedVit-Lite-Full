#!/bin/bash

set -e

SESSION1=false
SESSION2=false
BASELINE_ONLY=false
MEDVIT_ONLY=false

for arg in "$@"; do
    case $arg in
        --session1)      SESSION1=true ;;
        --session2)      SESSION2=true ;;
        --baseline-only) BASELINE_ONLY=true ;;
        --medvit-only)   MEDVIT_ONLY=true ;;
    esac
done

# Si aucun flag → lancer les deux sessions
if [ "$SESSION1" = false ] && [ "$SESSION2" = false ]; then
    SESSION1=true
    SESSION2=true
fi

CONFIG="configs/kaggle.yaml"
LOG_DIR="/kaggle/working/logs"
CKPT_DIR="/kaggle/working/checkpoints"

mkdir -p $LOG_DIR $CKPT_DIR /kaggle/working/results

echo "=============================================="
echo "  MedViT-Lite — Kaggle T4 x2"
echo "  $(date)"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ══════════════════════════════════════════════
#  SESSION 1 : Baseline + MedViT-Lite complet
# ══════════════════════════════════════════════
if [ "$SESSION1" = true ] && [ "$BASELINE_ONLY" = false ] || [ "$BASELINE_ONLY" = true ]; then

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [1/5] Baseline : ResNet-50"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/baseline_cnn.py \
        --config $CONFIG --no-wandb \
        2>&1 | tee $LOG_DIR/baseline_resnet50.log
    echo "✅ ResNet-50 terminé — checkpoint sauvé dans $CKPT_DIR"
fi

if [ "$BASELINE_ONLY" = true ]; then exit 0; fi

if [ "$SESSION1" = true ] && [ "$MEDVIT_ONLY" = false ] || [ "$MEDVIT_ONLY" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [2/5] MedViT-Lite complet (DPS + SFC + HTA)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/medvit_lite_train.py \
        --config $CONFIG --no-wandb \
        2>&1 | tee $LOG_DIR/medvit_lite_full.log
    echo "✅ MedViT-Lite complet terminé"
fi

if [ "$MEDVIT_ONLY" = true ]; then exit 0; fi

# ══════════════════════════════════════════════
#  SESSION 2 : Ablations
# ══════════════════════════════════════════════
if [ "$SESSION2" = true ]; then

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [3/5] Ablation : sans DPS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/medvit_lite_train.py \
        --config $CONFIG --no-wandb --no-dps \
        2>&1 | tee $LOG_DIR/medvit_lite_noDPS.log
    echo "✅ Ablation sans DPS terminée"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [4/5] Ablation : sans SFC"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/medvit_lite_train.py \
        --config $CONFIG --no-wandb --no-sfc \
        2>&1 | tee $LOG_DIR/medvit_lite_noSFC.log
    echo "✅ Ablation sans SFC terminée"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [5/5] Ablation : ViT de base (sans innovations)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/medvit_lite_train.py \
        --config $CONFIG --no-wandb --no-dps --no-sfc \
        2>&1 | tee $LOG_DIR/medvit_lite_noInnovations.log
    echo "✅ Ablation sans innovations terminée"
fi

echo ""
echo "=============================================="
echo "  Toutes les expériences terminées !"
echo "  $(date)"
echo "=============================================="
