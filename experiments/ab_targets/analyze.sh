#!/bin/bash
# Head-to-head match + loss-trend comparison for the two A/B arms.
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

AB="$(cd "$(dirname "$0")" && pwd)"
PY=$ROOT/venv/bin/python
cd $ROOT/python_src
cp "$AB/base/latest.pth" "$AB/base_final.pth"
cp "$AB/new/latest.pth"  "$AB/new_final.pth"
$PY -u eval.py "$AB/base_final.pth" "$AB/new_final.pth" \
    --games "${GAMES:-200}" --simulations "${SIMS:-100}" --max-moves 150 \
    --mcts-batch 50 --temperature-moves 8 --compile-inference none \
    2>&1 | tee "$AB/match.log"
