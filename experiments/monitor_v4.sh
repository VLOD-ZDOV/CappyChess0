#!/bin/bash
# Следит, что сеть умнеет, а не тупеет.
#   policy_health — не расплющивается ли голова обратно (быстро, каждый чекпоинт)
#   матч против стартового par8_boot — единственная настоящая мера силы
#     (каждый третий чекпоинт: матч отнимает карту у обучения)
# Пути выводятся из расположения скрипта — в репозитории не должно быть
# абсолютных путей с чужой файловой системы.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

cd $ROOT/python_src || exit 1
PY=../venv/bin/python
BASE=../experiments/ab_parallel/par8_boot.pth
OUT=../experiments/v4_health.log
CK=checkpoints_v4
n=0
seen=""
TPID=$(cat ../experiments/train_v4.pid)
# Проверяем по PID, а не по шаблону: 2026-09-08 шаблон совпал с зависшим
# `tmux send-keys`, в командной строке которого была та же строка запуска,
# и наблюдение продолжало следить за призраком после смерти обучения.
while kill -0 "$TPID" 2>/dev/null; do
  for f in $(ls $CK/model_iter*.pth 2>/dev/null); do
    case "$seen" in *"$f"*) continue;; esac
    seen="$seen $f"
    n=$((n+1))
    {
      echo "=============================================================="
      echo "$(date +%H:%M)  $f"
      $PY policy_health.py "$BASE" "$f" 2>/dev/null | grep -avE "RuntimeWarning|frozen"
    } >> $OUT
    if [ $((n % 3)) -eq 0 ]; then
      {
        echo "-- матч против стартового par8_boot, 200 партий --"
        $PY eval.py "$BASE" "$f" --games 200 --simulations 200 \
            --mcts-batch 64 --mcts-parallel-sims 8 2>/dev/null \
            | grep -aE "Итог:"
      } >> $OUT
    fi
  done
  sleep 60
done
echo "$(date +%H:%M)  обучение завершилось, наблюдение остановлено" >> $OUT
