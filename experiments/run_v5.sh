#!/bin/bash
# Прогон v5: перезапуск v4 с ПОЧИНЕННЫМ чекпоинтом.
#
# v4 провалился (-96 Elo) не из-за настроек поиска, а из-за того, что
# bootstrap_policy.py оставлял в чекпоинте старую EMA. train.py делает
# ema.apply_to(net) перед self-play, поэтому партии играла отстающая на 18%
# сеть, и починка головы не доезжала до данных. Исправлено в инструменте,
# par8_boot.pth перезаписан (расхождение EMA с model теперь 0.0000).
#
# games/mcts-batch снижены со 128 до 64, train-steps со 200 до 100:
# соотношение шагов на свежую позицию сохранено (0.05), а нагрузка на карту
# вдвое меньше — чтобы не рвать 1% low в игре на той же видеокарте.
# nice 19: обучение уступает CPU игре, планировщик кадров важнее.
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

cd $ROOT/python_src || exit 1
echo $$ > ../experiments/train_v5.pid
exec nice -n 19 ../venv/bin/python -u train.py \
  --channels 128 --res-blocks 10 --transformer-blocks 2 --transformer-heads 8 \
  --attn-policy \
  --games 64 --mcts-batch 64 --mcts-parallel-sims 8 \
  --simulations 600 --fast-simulations 150 \
  --batch-size 512 --train-steps 100 \
  --buffer-max 150000 --buffer-min-to-train 12000 \
  --lr 2e-4 --save-every 10 --max-iters 60 \
  --reset-buffer --reset-scheduler \
  --checkpoint-dir checkpoints_v5
