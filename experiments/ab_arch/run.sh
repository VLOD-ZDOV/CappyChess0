#!/bin/bash
# A/B: legacy heads/FFN vs the modern transformer stack, same trunk.
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

set -e
AB="$(cd "$(dirname "$0")" && pwd)"
PY=$ROOT/venv/bin/python
cd $ROOT/python_src
COMMON="--channels 64 --res-blocks 6 --transformer-blocks 2 --transformer-heads 4
        --games 64 --mcts-batch 64 --mcts-parallel-sims 32
        --simulations 100 --fast-simulations 50
        --batch-size 256 --train-steps 100 --min-train-steps 100
        --buffer-min-to-train 2000 --buffer-max 200000
        --max-game-length 120 --lr 1e-3 --save-every 999 --log-every 100
        --max-iters ${ITERS:-40}"
$PY -u train.py $COMMON --checkpoint-dir "$AB/legacy" > "$AB/legacy.log" 2>&1
$PY -u train.py $COMMON --checkpoint-dir "$AB/modern" \
    --attn-policy --qk-norm --swiglu --registers 4 --value-residual \
    --no-qkv-bias --rmsnorm > "$AB/modern.log" 2>&1
echo "both arms done"
