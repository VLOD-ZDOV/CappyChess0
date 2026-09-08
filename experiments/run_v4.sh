#!/bin/bash
# Прогон v4: от bootstrap-чекпоинта, 600 симуляций.
#
# v3 (100 симуляций) пришлось остановить: голова расплющивалась обратно,
# max(p)*n 4.31 -> 2.35 за 26 итераций. Причина не в резкости цели, а в её
# ТОЧНОСТИ: цель на 100 симуляциях указывает лучший ход эталона лишь в 21.9%
# случаев против 53.1% у целей, на которых голову чинили. Сеть не подгоняется
# под шум и усредняет. См. docs/experiments/policy_target_collapse.md.
#
# 600/8 = 75 последовательных раундов PUCT, быстрый поиск PCR 150/8 = 19.
#
# Запуск ТОЛЬКО через setsid, вне tmux: 2026-09-08 tmux-сервер умер и утащил
# обучение с собой.
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

cd $ROOT/python_src || exit 1
# exec сохраняет pid, поэтому $$ здесь — это и есть pid обучения.
# pgrep по шаблону ловил транзиентную обёртку и давал не тот номер.
echo $$ > ../experiments/train_v4.pid

exec ../venv/bin/python -u train.py \
  --channels 128 --res-blocks 10 --transformer-blocks 2 --transformer-heads 8 \
  --attn-policy \
  --games 128 --mcts-batch 128 --mcts-parallel-sims 8 \
  --simulations 600 --fast-simulations 150 \
  --batch-size 512 --train-steps 200 \
  --buffer-max 150000 --buffer-min-to-train 12000 \
  --lr 2e-4 --save-every 10 --max-iters 60 \
  --reset-buffer --reset-scheduler \
  --checkpoint-dir checkpoints_v4
