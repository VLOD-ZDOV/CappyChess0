#!/bin/bash
# Обучение от bootstrap-чекпоинта с исправленными настройками.
# parallel 8 (было 32) — см. docs/experiments/policy_target_collapse.md
#
# Запускается ЧЕРЕЗ setsid, вне tmux: 2026-09-08 tmux-сервер умер и утащил
# обучение с собой. tmux теперь только показывает лог, но ничего не держит.
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

cd $ROOT/python_src || exit 1
# exec сохраняет pid, поэтому $$ здесь — это и есть pid обучения.
# pgrep по шаблону ловил транзиентную обёртку и давал не тот номер.
echo $$ > ../experiments/train_v3.pid

exec ../venv/bin/python -u train.py \
  --channels 128 --res-blocks 10 --transformer-blocks 2 --transformer-heads 8 \
  --attn-policy \
  --games 128 --mcts-batch 128 --mcts-parallel-sims 8 \
  --simulations 100 --fast-simulations 50 \
  --batch-size 512 --train-steps 200 \
  --buffer-max 150000 --buffer-min-to-train 20000 \
  --lr 2e-4 --save-every 10 --max-iters 90 \
  --checkpoint-dir checkpoints_v3
