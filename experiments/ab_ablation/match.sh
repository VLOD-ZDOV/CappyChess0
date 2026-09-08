#!/bin/bash
# Each ablated arm plays the FULL bundle. 50% => that flag contributes nothing.
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

AB="$(cd "$(dirname "$0")" && pwd)"
PY=$ROOT/venv/bin/python
cd $ROOT/python_src
cp "$AB/full/latest.pth" "$AB/full_final.pth"
for arm in no_attnpol no_qknorm no_swiglu no_regs no_vres; do
  cp "$AB/$arm/latest.pth" "$AB/${arm}_final.pth"
  $PY -u eval.py "$AB/full_final.pth" "$AB/${arm}_final.pth" \
      --games "${GAMES:-200}" --simulations 100 --max-moves 150 \
      --mcts-batch 50 --temperature-moves 8 --compile-inference none \
      > "$AB/match_$arm.log" 2>&1
  echo "=== full vs $arm ==="; grep "Итог:" "$AB/match_$arm.log"
done
echo "all matches done"
