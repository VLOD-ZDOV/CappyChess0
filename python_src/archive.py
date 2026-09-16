"""Архив самоигры: хранить всё и уметь спрашивать.

Зачем отдельно от буфера обучения. Буфер — кольцо на 400k позиций, он забывает
старое и держит только то, на чём учатся ПРЯМО СЕЙЧАС: позиции полного поиска,
с уже перемешанной целью value. Архив решает другую задачу — сохранить факты,
из которых потом можно собрать любой набор: партии с матом, победы чёрных,
короткие разгромы, позиции быстрого поиска. Поэтому здесь лежат исходные
величины (исход партии, оценка корня), а не готовые цели обучения: цель
пересобирается, исход — нет.

Формат — два файла на итерацию:

  boards_000291.f16   доски подряд, сырой float16, без заголовка
  meta_000291.npz     все остальные столбцы, каждый маленький

Доски отдельным сырым файлом ради двух вещей. Первое: его можно
`np.memmap` без разбора zip, то есть читать выборку из архива, который больше
ОЗУ — в память попадают только запрошенные строки. Второе: он пишется
дописыванием по ходу итерации, поэтому пиковая память не растёт с числом
партий. На ZFS с zstd такой файл ужимается примерно в 65 раз: доска почти
целиком нули.

Единица запроса — ПАРТИЯ. Отбор идёт по её свойствам (исход, длина, чем
кончилась), а позиции берутся уже у отобранных партий.
"""
import os
import re
import numpy as np

BOARDS_RE = re.compile(r"^boards_(\d+)\.f16$")

# Чем кончилась партия.
TERM_MATE = 0
TERM_RESIGN = 1
TERM_DRAW_RULE = 2      # пат, 50 ходов, недостаточный материал
TERM_LIMIT = 3          # упёрлась в max_game_length, исход по материалу
TERM_ADJUDICATED = 4    # судья по материалу
TERM_NAMES = {TERM_MATE: "мат", TERM_RESIGN: "сдача", TERM_DRAW_RULE: "ничья по правилам",
              TERM_LIMIT: "лимит ходов", TERM_ADJUDICATED: "судья"}

# Столбцы позиции и их типы. Держим списком, чтобы читатель и писатель не
# разъезжались.
POS_COLS = {
    "game": np.int32,     # в какую партию входит (индекс в таблице партий)
    "ply": np.int16,      # номер полухода с начала партии, с нуля
    "side": np.int8,      # кто ходит: 0 белые, 1 чёрные
    "full": np.int8,      # 1 полный поиск, 0 быстрый (playout cap)
    "root_q": np.float16, # оценка корня поиском, с точки зрения ходящего
    "root_d": np.float16, # вероятность ничьей по поиску (-1 = не было)
}
GAME_COLS = {
    "result": np.int8,    # +1 победа белых, 0 ничья, -1 победа чёрных
    "plies": np.int16,    # длина партии в полуходах
    "term": np.int8,      # TERM_*
    "iter": np.int32,     # итерация обучения
    "playthrough": np.int8,   # 1 = партия игралась без права сдаться
    # Полуход, на котором сдача сработала бы: НОМЕР ПОЗИЦИИ, тот же, что в поле
    # `ply` позиций, — чтобы можно было брать позиции после приговора одним
    # сравнением. -1 = приговора не было.
    "resign_ply": np.int16,
    # Кто сдавался. Выводить из чётности полухода нельзя: счётчик в самоигре
    # увеличивается ДО проверки порога, и чётность оказывается сдвинута.
    "resign_side": np.int8,   # 0 белые, 1 чёрные, -1 приговора не было
}


class ArchiveWriter:
    """Пишет одну итерацию. Доски дописываются сразу, столбцы копятся в списках
    (они мелкие) и уходят в meta одним куском в close()."""

    def __init__(self, directory, iteration, board_len):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.iteration = int(iteration)
        self.board_len = int(board_len)
        self._boards_path = os.path.join(directory, f"boards_{self.iteration:06d}.f16")
        self._tmp = self._boards_path + ".tmp"
        self._f = open(self._tmp, "wb")
        self.pos = {k: [] for k in POS_COLS}
        self.games = {k: [] for k in GAME_COLS}
        self.pol_idx = []
        self.pol_val = []
        self.rows = 0

    def add_game(self, result, plies, term, playthrough, resign_ply, resign_side=-1):
        """Завести партию и вернуть её индекс — его кладут в поле `game` позиций."""
        self.games["result"].append(int(result))
        self.games["plies"].append(int(plies))
        self.games["term"].append(int(term))
        self.games["iter"].append(self.iteration)
        self.games["playthrough"].append(int(bool(playthrough)))
        self.games["resign_ply"].append(int(resign_ply))
        self.games["resign_side"].append(int(resign_side))
        return len(self.games["result"]) - 1

    def add_position(self, board, pol_idx, pol_val, game, ply, side, full,
                     root_q, root_d):
        b = np.asarray(board, dtype=np.float16)
        if b.size != self.board_len:
            raise ValueError(f"доска {b.size} элементов, ожидалось {self.board_len}")
        self._f.write(b.tobytes())
        self.rows += 1
        self.pol_idx.append(np.asarray(pol_idx, dtype=np.int16))
        self.pol_val.append(np.asarray(pol_val, dtype=np.float16))
        self.pos["game"].append(game)
        self.pos["ply"].append(ply)
        self.pos["side"].append(side)
        self.pos["full"].append(1 if full else 0)
        self.pos["root_q"].append(root_q)
        self.pos["root_d"].append(-1.0 if root_d is None else root_d)

    def close(self):
        """Дописать meta и сделать файлы видимыми. Пустая итерация не
        оставляет за собой ничего."""
        self._f.close()
        if self.rows == 0:
            os.remove(self._tmp)
            return 0
        n = self.rows
        max_k = max((len(a) for a in self.pol_idx), default=0)
        pi = np.full((n, max_k), -1, dtype=np.int16)
        pv = np.zeros((n, max_k), dtype=np.float16)
        for i, (a, b) in enumerate(zip(self.pol_idx, self.pol_val)):
            pi[i, :len(a)] = a
            pv[i, :len(b)] = b
        out = {"pol_idx": pi, "pol_val": pv,
               "board_len": np.asarray([self.board_len], dtype=np.int32)}
        for k, dt in POS_COLS.items():
            out[k] = np.asarray(self.pos[k], dtype=dt)
        for k, dt in GAME_COLS.items():
            out["g_" + k] = np.asarray(self.games[k], dtype=dt)
        meta = os.path.join(self.dir, f"meta_{self.iteration:06d}.npz")
        tmp_meta = meta + ".tmp.npz"        # savez сам добавляет .npz, если его нет
        np.savez(tmp_meta, **out)
        # Сначала meta, потом доски: читатель перечисляет куски по boards_*, и
        # пока этого файла нет, незаконченная пара ему не попадётся.
        os.replace(tmp_meta, meta)
        os.replace(self._tmp, self._boards_path)
        return n


class Archive:
    """Чтение и отбор. Все столбцы держатся в памяти (их около 20 байт на
    позицию — миллион позиций это 20 МБ), доски не читаются вовсе, пока их не
    попросят: `boards()` берёт только выбранные строки через memmap."""

    def __init__(self, directory):
        self.dir = directory
        self.shards = []            # (итерация, путь к доскам, meta)
        games = {k: [] for k in GAME_COLS}
        pos = {k: [] for k in POS_COLS}
        shard_id, row_in_shard, game_off = [], [], 0
        names = sorted(n for n in os.listdir(directory) if BOARDS_RE.match(n)) \
            if os.path.isdir(directory) else []
        for si, name in enumerate(names):
            it = int(BOARDS_RE.match(name).group(1))
            meta_path = os.path.join(directory, f"meta_{it:06d}.npz")
            if not os.path.exists(meta_path):
                continue
            with np.load(meta_path) as z:
                m = {k: z[k] for k in z.files}
            self.shards.append((it, os.path.join(directory, name), m))
            n = m["game"].shape[0]
            for k in POS_COLS:
                pos[k].append(m[k])
            for k in GAME_COLS:
                games[k].append(m["g_" + k])
            # Индексы партий локальны для куска — сдвигаем в сквозную нумерацию.
            pos["game"][-1] = pos["game"][-1].astype(np.int64) + game_off
            game_off += m["g_result"].shape[0]
            shard_id.append(np.full(n, len(self.shards) - 1, dtype=np.int32))
            row_in_shard.append(np.arange(n, dtype=np.int64))
        if not self.shards:
            raise FileNotFoundError(f"в {directory} нет ни одного куска архива")
        self.g = {k: np.concatenate(v) for k, v in games.items()}
        self.p = {k: np.concatenate(v) for k, v in pos.items()}
        self._shard = np.concatenate(shard_id)
        self._row = np.concatenate(row_in_shard)
        self.board_len = int(self.shards[0][2]["board_len"][0])

    # ── отбор ────────────────────────────────────────────────────────────────

    def games_where(self, result=None, term=None, min_plies=None, max_plies=None,
                    plies=None, moves=None, playthrough=None, iters=None,
                    false_resign=None, decisive=None):
        """Маска по партиям.

        result      +1 / 0 / -1 либо список
        term        TERM_* либо список
        moves       номер ПОЛНОГО хода, на котором партия кончилась (34 → 67-68
                    полуходов); можно диапазон (a, b)
        plies       точная длина в полуходах, число или (a, b)
        decisive    True — только результативные
        false_resign True — доигранные партии, где приговор о сдаче оказался
                    неверным (сторона не проиграла)
        """
        m = np.ones(self.g["result"].shape[0], dtype=bool)
        def _in(col, val):
            nonlocal m
            if val is None:
                return
            v = np.atleast_1d(np.asarray(val))
            m &= np.isin(self.g[col], v)
        _in("result", result)
        _in("term", term)
        _in("iter", iters)
        if playthrough is not None:
            m &= self.g["playthrough"] == int(bool(playthrough))
        if decisive:
            m &= self.g["result"] != 0
        if min_plies is not None:
            m &= self.g["plies"] >= min_plies
        if max_plies is not None:
            m &= self.g["plies"] <= max_plies
        if plies is not None:
            lo, hi = plies if isinstance(plies, (tuple, list)) else (plies, plies)
            m &= (self.g["plies"] >= lo) & (self.g["plies"] <= hi)
        if moves is not None:
            lo, hi = moves if isinstance(moves, (tuple, list)) else (moves, moves)
            # Ход n — это полуходы 2n-1 и 2n.
            m &= (self.g["plies"] >= 2 * lo - 1) & (self.g["plies"] <= 2 * hi)
        if false_resign:
            # Приговор верен, если приговорённая сторона и правда проиграла.
            side = self.g["resign_side"]
            verdict_ok = np.where(side == 0, self.g["result"] < 0,
                                  self.g["result"] > 0)
            m &= ((self.g["playthrough"] == 1) & (self.g["resign_ply"] >= 0)
                  & (~verdict_ok))
        return m

    def positions_where(self, game_mask=None, full=None, side=None,
                        min_ply=None, max_ply=None, outcome=None):
        """Маска по позициям. `outcome` — исход с точки зрения ТОГО, КТО ХОДИТ:
        "win" / "loss" / "draw"; так отбираются «выигранные»/«проигранные»
        позиции независимо от цвета."""
        m = np.ones(self.p["game"].shape[0], dtype=bool)
        if game_mask is not None:
            m &= game_mask[self.p["game"]]
        if full is not None:
            m &= self.p["full"] == int(bool(full))
        if side is not None:
            m &= self.p["side"] == int(side)
        if min_ply is not None:
            m &= self.p["ply"] >= min_ply
        if max_ply is not None:
            m &= self.p["ply"] <= max_ply
        if outcome is not None:
            r = self.g["result"][self.p["game"]]
            pov = np.where(self.p["side"] == 0, r, -r)   # исход глазами ходящего
            m &= {"win": pov > 0, "loss": pov < 0, "draw": pov == 0}[outcome]
        return m

    # ── чтение ───────────────────────────────────────────────────────────────

    def boards(self, mask, batch=8192):
        """Доски выбранных позиций, (k, board_len) float16. Читается через
        memmap по кускам: в ОЗУ оказывается только результат, сам архив может
        быть сколь угодно больше памяти."""
        idx = np.flatnonzero(mask)
        out = np.empty((idx.size, self.board_len), dtype=np.float16)
        for si in np.unique(self._shard[idx]):
            sel = idx[self._shard[idx] == si]
            path = self.shards[si][1]
            rows = os.path.getsize(path) // (2 * self.board_len)
            mm = np.memmap(path, dtype=np.float16, mode="r",
                           shape=(rows, self.board_len))
            local = self._row[sel]
            order = np.argsort(local)          # последовательное чтение с диска
            dest = np.flatnonzero(np.isin(idx, sel))
            for a in range(0, order.size, batch):
                sl = order[a:a + batch]
                out[dest[sl]] = mm[local[sl]]
            del mm
        return out

    def policy(self, mask):
        """Разреженная политика выбранных позиций: список пар (индексы, веса)."""
        idx = np.flatnonzero(mask)
        out = []
        for si in np.unique(self._shard[idx]):
            sel = idx[self._shard[idx] == si]
            m = self.shards[si][2]
            for r in self._row[sel]:
                pi, pv = m["pol_idx"][r], m["pol_val"][r]
                k = pi >= 0
                out.append((pi[k], pv[k]))
        return out

    def summary(self):
        n_g = self.g["result"].shape[0]
        n_p = self.p["game"].shape[0]
        lines = [f"архив: {n_g:,} партий, {n_p:,} позиций, "
                 f"итерации {self.g['iter'].min()}–{self.g['iter'].max()}"]
        for t, name in TERM_NAMES.items():
            c = int((self.g["term"] == t).sum())
            if c:
                lines.append(f"  {name}: {c:,} ({100.0*c/n_g:.1f}%)")
        full = int((self.p["full"] == 1).sum())
        lines.append(f"  позиций полного поиска {full:,}, быстрых {n_p-full:,}")
        return "\n".join(lines).replace(",", " ")
