#!/bin/bash

set -eo pipefail   # -e = stop on error, -o pipefail = catch errors through pipes
export MALLOC_TRIM_THRESHOLD_=65536

SESSION1=false
SESSION2=false
BASELINE_ONLY=false
MEDVIT_ONLY=false
DRY_RUN=false
RESUME=false
USE_WANDB=false

EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --session1)      SESSION1=true ;;
        --session2)      SESSION2=true ;;
        --baseline-only) BASELINE_ONLY=true ;;
        --medvit-only)   MEDVIT_ONLY=true ;;
        --dry-run)       DRY_RUN=true ;;
        --resume)        RESUME=true ;;
        --with-wandb)    USE_WANDB=true ;;
    esac
done

# Construire les arguments supplémentaires
if [ "$DRY_RUN" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --dry-run"
fi

if [ "$RESUME" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume"
fi

if [ "$USE_WANDB" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no-wandb"
fi

# Si aucun flag de session → lancer les deux sessions
if [ "$SESSION1" = false ] && [ "$SESSION2" = false ]; then
    SESSION1=true
    SESSION2=true
fi

# Vérifier et installer les dépendances si nécessaire
if ! python -c "import medmnist, timm, einops, rich" &> /dev/null; then
    echo "📦 Installation des dépendances manquantes (medmnist, timm, einops, rich)..."
    pip install -q -r requirements.txt || pip install -q medmnist timm einops rich wandb pyyaml
    echo "✅ Dépendances installées avec succès."
fi

CONFIG="configs/base.yaml"   # base.yaml contient déjà la config Kaggle
LOG_DIR="/kaggle/working/logs"
CKPT_DIR="/kaggle/working/checkpoints"
RESULTS_DIR="/kaggle/working/results"

mkdir -p $LOG_DIR $CKPT_DIR $RESULTS_DIR

echo "=============================================="
echo "  MedViT-Lite — Entraînement Kaggle"
echo "  $(date)"
if [ "$DRY_RUN" = true ]; then
    echo "  [MODE TEST DRY-RUN : 2 epochs rapides]"
fi
if [ "$RESUME" = true ]; then
    echo "  [MODE REPRISE / RESUME ACTIVÉ]"
fi
echo "=============================================="

if command -v nvidia-smi &> /dev/null; then
    echo "GPUs détectés :"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "⚠️ ATTENTION : nvidia-smi introuvable. Exécution sur CPU possiblement lente."
fi
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
        --config $CONFIG $EXTRA_ARGS \
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
        --config $CONFIG $EXTRA_ARGS \
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
        --config $CONFIG $EXTRA_ARGS --no-dps \
        2>&1 | tee $LOG_DIR/medvit_lite_noDPS.log
    echo "✅ Ablation sans DPS terminée"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [4/5] Ablation : sans SFC"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/medvit_lite_train.py \
        --config $CONFIG $EXTRA_ARGS --no-sfc \
        2>&1 | tee $LOG_DIR/medvit_lite_noSFC.log
    echo "✅ Ablation sans SFC terminée"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [5/5] Ablation : ViT de base (sans innovations)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python experiments/medvit_lite_train.py \
        --config $CONFIG $EXTRA_ARGS --no-dps --no-sfc \
        2>&1 | tee $LOG_DIR/medvit_lite_noInnovations.log
    echo "✅ Ablation sans innovations terminée"
fi

echo ""
echo "=============================================="
echo "  Génération du rapport de comparaison"
echo "=============================================="
if [ -f experiments/compare_results.py ]; then
    python experiments/compare_results.py --results-dir $RESULTS_DIR || true
fi

echo ""
echo "=============================================="
echo "  Toutes les expériences terminées !"
echo "  $(date)"
echo "=============================================="
