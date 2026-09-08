#!/bin/bash
# Leave-one-out ablation of the modern bundle.
# Each arm is the FULL bundle minus one flag, so the match against the full
# bundle measures exactly what that flag contributes.
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
        --seed 1234 --max-iters ${ITERS:-40}"
FULL="--attn-policy --qk-norm --swiglu --registers 4 --value-residual --no-qkv-bias --rmsnorm"

run () {   # name, flags...
  local name=$1; shift
  mkdir -p "$AB/$name"
  $PY -u train.py $COMMON "$@" --checkpoint-dir "$AB/$name" > "$AB/$name.log" 2>&1
  echo "arm done: $name"
}

run full        $FULL
run no_attnpol  --qk-norm --swiglu --registers 4 --value-residual --no-qkv-bias --rmsnorm
run no_qknorm   --attn-policy --swiglu --registers 4 --value-residual --no-qkv-bias --rmsnorm
run no_swiglu   --attn-policy --qk-norm --registers 4 --value-residual --no-qkv-bias --rmsnorm
run no_regs     --attn-policy --qk-norm --swiglu --value-residual --no-qkv-bias --rmsnorm
run no_vres     --attn-policy --qk-norm --swiglu --registers 4 --no-qkv-bias --rmsnorm
echo "all arms done"
