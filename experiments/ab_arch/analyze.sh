#!/bin/bash
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

AB="$(cd "$(dirname "$0")" && pwd)"
PY=$ROOT/venv/bin/python
cd $ROOT/python_src
cp "$AB/legacy/latest.pth" "$AB/legacy_final.pth"
cp "$AB/modern/latest.pth" "$AB/modern_final.pth"
$PY -u eval.py "$AB/legacy_final.pth" "$AB/modern_final.pth" \
    --games "${GAMES:-200}" --simulations "${SIMS:-100}" --max-moves 150 \
    --mcts-batch 50 --temperature-moves 8 --compile-inference none \
    2>&1 | tee "$AB/match.log"
