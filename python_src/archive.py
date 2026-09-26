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

BOARDS_RE = re.compile(r"^boards_(\d+)\.(f16|fb)$")

# Упакованные доски (.fb): все плоскости кодировщика бинарные, кроме счётчика
# полуходов (halfmove/100). Строка = биты бинарных плоскостей + 80 чисел float16
# этой плоскости: 1540 байт вместо 22 240. Строки, где «бинарная» плоскость
# всё-таки не 0/1 (или -0.0), лежат целиком в boards_N.fb.raw.npz — без потерь.
_SQ = 80
_HM0, _HM1 = (8 * 17 + 1) * _SQ, (8 * 17 + 2) * _SQ
_F16_ONE = int(np.float16(1.0).view(np.uint16))
FB_BOARD_LEN = 139 * _SQ                # упаковка знает только эту кодировку


def _fb_layout(board_len):
    nbits = board_len - _SQ
    nbytes = (nbits + 7) // 8
    return nbits, nbytes, nbytes + 2 * _SQ


def pack_boards(b):
    """(m, board_len) float16 → (строки uint8 (m, stride), номера строк-исключений)."""
    b = np.ascontiguousarray(b, dtype=np.float16)
    u = b.view(np.uint16)
    binary = np.concatenate([u[:, :_HM0], u[:, _HM1:]], axis=1)
    ok = ((binary == 0) | (binary == _F16_ONE)).all(axis=1)
    bits = np.packbits(binary != 0, axis=1)
    hm = np.ascontiguousarray(b[:, _HM0:_HM1]).view(np.uint8)
    return np.concatenate([bits, hm], axis=1), np.flatnonzero(~ok)


def unpack_boards(rows_u8, board_len):
    nbits, nbytes, _ = _fb_layout(board_len)
    bits = np.unpackbits(rows_u8[:, :nbytes], axis=1, count=nbits)
    out = np.empty((rows_u8.shape[0], board_len), dtype=np.float16)
    out[:, :_HM0] = bits[:, :_HM0]
    out[:, _HM0:_HM1] = np.ascontiguousarray(rows_u8[:, nbytes:]).view(np.float16)
    out[:, _HM1:] = bits[:, _HM0:]
    return out


class BoardFile:
    """Доски одного куска по индексу строки, в любом формате: bf[i] или bf[[i, j]]."""

    def __init__(self, meta_path, board_len):
        base = meta_path.replace("meta_", "boards_")[:-len(".npz")]
        self.path = next((base + e for e in (".fb", ".f16") if os.path.exists(base + e)), None)
        if self.path is None:
            raise FileNotFoundError(base)
        self.board_len = board_len

    def __getitem__(self, i):
        if np.isscalar(i):
            return read_board_rows(self.path, [int(i)], self.board_len)[0]
        return read_board_rows(self.path, i, self.board_len)


def read_board_rows(path, rows, board_len):
    """Доски строк `rows` из файла кусков любого формата (.f16 или .fb)."""
    rows = np.asarray(rows, dtype=np.int64)
    if path.endswith(".f16"):
        n = os.path.getsize(path) // (2 * board_len)
        mm = np.memmap(path, dtype=np.float16, mode="r", shape=(n, board_len))
        return np.asarray(mm[rows])
    stride = _fb_layout(board_len)[2]
    n = os.path.getsize(path) // stride
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, stride))
    out = unpack_boards(np.asarray(mm[rows]), board_len)
    raw = path + ".raw.npz"
    if os.path.exists(raw):
        with np.load(raw) as z:
            where = {int(r): i for i, r in enumerate(z["rows"])}
            for j, r in enumerate(rows):
                i = where.get(int(r))
                if i is not None:
                    out[j] = z["boards"][i]
    return out

# Чем кончилась партия.
TERM_MATE = 0
TERM_RESIGN = 1
TERM_DRAW_RULE = 2      # ничья по правилам без уточнения (старые куски архива)
TERM_LIMIT = 3          # упёрлась в max_game_length, исход по материалу
TERM_ADJUDICATED = 4    # судья по материалу
# Ничьи разделены: «пат» и «50 ходов» — совершенно разные вещи. Пат означает,
# что сторона загнала соперника в позицию без ходов и упустила выигрыш; 50 ходов
# — что не смогла реализовать перевес; недостаток материала — что реализовывать
# было нечем. В журнале до 16.09 все три печатались одним словом «пат».
TERM_STALEMATE = 5
TERM_FIFTY = 6
TERM_REPETITION = 7
TERM_MATERIAL = 8
TERM_NAMES = {TERM_MATE: "мат", TERM_RESIGN: "сдача",
              TERM_DRAW_RULE: "ничья по правилам (без уточнения)",
              TERM_LIMIT: "лимит ходов", TERM_ADJUDICATED: "судья",
              TERM_STALEMATE: "пат", TERM_FIFTY: "50 ходов",
              TERM_REPETITION: "троекратное повторение",
              TERM_MATERIAL: "недостаток материала"}
# draw_reason() в движке → TERM_*
DRAW_REASON_TO_TERM = {0: TERM_STALEMATE, 1: TERM_FIFTY,
                       2: TERM_REPETITION, 3: TERM_MATERIAL}

SOURCE_SELFPLAY = 0
SOURCE_MATCH = 1        # сеть против сети, 800 симуляций, сдачи нет
SOURCE_FSF = 2          # сеть против движка
SOURCE_HUMAN = 3        # партия человека, загружена из PGN — политики нет
# Позиция из полного перебора дебюта: это НЕ партия, исхода у неё нет, в
# root_q лежит оценка движка. Из статистики книги такие записи исключаются,
# иначе 800 тысяч мнимых ничьих затопят настоящие исходы.
SOURCE_BOOK = 4
SOURCE_NAMES = {0: "самоигра", 1: "матч сетей", 2: "против движка",
                3: "человек", 4: "перебор дебюта"}

# Столбцы позиции и их типы. Держим списком, чтобы читатель и писатель не
# разъезжались.
POS_COLS = {
    "game": np.int32,     # в какую партию входит (индекс в таблице партий)
    "ply": np.int16,      # номер полухода с начала партии, с нуля
    "side": np.int8,      # кто ходит: 0 белые, 1 чёрные
    "full": np.int8,      # 1 полный поиск, 0 быстрый (playout cap)
    "root_q": np.float16, # оценка корня поиском, с точки зрения ходящего
    "root_d": np.float16, # вероятность ничьей по поиску (-1 = не было)
    # Сделанный ход, канонический индекс политики (-1 = неизвестен). Без него
    # архив неполон: из него нельзя ни восстановить партию, ни собрать цель
    # future-головы, которой нужен ход через два полухода.
    "move": np.int16,
    # Тот же ход, но в координатах ДОСКИ: (откуда << 10) | (куда << 3) | превращение.
    # Индекс политики для чёрных считается по перевёрнутой доске, поэтому назвать
    # по нему ход человеческим языком нельзя — а по этому можно.
    "move_raw": np.int32,
}
GAME_COLS = {
    "result": np.int8,    # +1 победа белых, 0 ничья, -1 победа чёрных
    "plies": np.int16,    # длина партии в полуходах
    "term": np.int8,      # TERM_*
    "iter": np.int32,     # итерация обучения
    # Откуда партия: 0 самоигра, 1 матч между двумя сетями. У матчевых партий
    # другое качество — поиск глубже (800 симуляций против 600/100) и сдачи нет
    # вовсе, поэтому метка value не может оказаться ложной сдачей. Но они
    # OFF-POLICY: их играли другие чекпоинты, и в живой буфер их лить нельзя.
    "source": np.int8,
    # Кто играл каждой стороной — индекс в массиве `names` из meta (-1 неизвестно).
    "white_id": np.int8,
    "black_id": np.int8,
    "playthrough": np.int8,   # 1 = партия игралась без права сдаться
    # Полуход, на котором сдача сработала бы: НОМЕР ПОЗИЦИИ, тот же, что в поле
    # `ply` позиций, — чтобы можно было брать позиции после приговора одним
    # сравнением. -1 = приговора не было.
    "resign_ply": np.int16,
    # Кто сдавался. Выводить из чётности полухода нельзя: счётчик в самоигре
    # увеличивается ДО проверки порога, и чётность оказывается сдвинута.
    "resign_side": np.int8,   # 0 белые, 1 чёрные, -1 приговора не было
}


# Открытые писатели — чтобы обработчик сигнала мог их закрыть. Иначе убитый
# посреди итерации процесс оставлял boards_*.f16.tmp без meta, и вся сыгранная
# с начала итерации самоигра пропадала.
_OPEN = []


def close_open_writers():
    """Закрыть все архивы, открытые ЭТИМ процессом. Возвращает (файлов, строк).
    Зовётся из обработчика сигнала, поэтому ошибки глотает."""
    saved = rows = 0
    for w in list(_OPEN):
        if w._pid != os.getpid():
            continue
        try:
            n = w.close()
            if n:
                saved += 1
                rows += n
        except Exception:
            pass
    return saved, rows


class ArchiveWriter:
    """Пишет одну итерацию. Доски дописываются сразу, столбцы копятся в списках
    (они мелкие) и уходят в meta одним куском в close()."""

    def __init__(self, directory, iteration, board_len, names=None):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.iteration = int(iteration)
        self.board_len = int(board_len)
        # Архив общий для всех линий: номер итерации в имени файла может быть уже
        # занят другой линией, и os.replace молча затёр бы её партии. Номер в
        # имени — только ключ пары boards/meta; настоящая итерация лежит в meta.
        fid = self.iteration
        while any(os.path.exists(os.path.join(directory, f"{pre}_{fid:06d}{ext}"))
                  for pre, ext in (("boards", ".f16"), ("boards", ".f16.tmp"),
                                   ("boards", ".fb"), ("boards", ".fb.tmp"),
                                   ("meta", ".npz"))):
            fid += 10_000_000
        self.file_id = fid
        self._packed = self.board_len == FB_BOARD_LEN
        ext = ".fb" if self._packed else ".f16"
        self._boards_path = os.path.join(directory, f"boards_{fid:06d}{ext}")
        self._stride = _fb_layout(self.board_len)[2] if self._packed else 2 * self.board_len
        self._raw = {}                       # строка → доска целиком (редкие исключения)
        self._tmp = self._boards_path + ".tmp"
        self._f = open(self._tmp, "wb")
        self.pos = {k: [] for k in POS_COLS}
        self.games = {k: [] for k in GAME_COLS}
        self.pol_idx = []
        self.pol_val = []
        self.rows = 0
        self._pid = os.getpid()
        self._closed = False
        _OPEN.append(self)
        # Имена игроков (чекпоинтов). Хранятся строками один раз, в партиях
        # лежат индексы — иначе имя повторялось бы у каждой позиции.
        self.names = list(names or [])

    def name_id(self, name):
        """Индекс имени, добавляя его при первой встрече."""
        if name is None:
            return -1
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)

    def add_game(self, result, plies, term, playthrough, resign_ply, resign_side=-1,
                 source=SOURCE_SELFPLAY, white_id=-1, black_id=-1):
        """Завести партию и вернуть её индекс — его кладут в поле `game` позиций."""
        self.games["result"].append(int(result))
        self.games["plies"].append(int(plies))
        self.games["term"].append(int(term))
        self.games["iter"].append(self.iteration)
        self.games["playthrough"].append(int(bool(playthrough)))
        self.games["resign_ply"].append(int(resign_ply))
        self.games["resign_side"].append(int(resign_side))
        self.games["source"].append(int(source))
        self.games["white_id"].append(int(white_id))
        self.games["black_id"].append(int(black_id))
        return len(self.games["result"]) - 1

    def add_position(self, board, pol_idx, pol_val, game, ply, side, full,
                     root_q, root_d, move=-1, move_raw=-1):
        b = np.asarray(board, dtype=np.float16)
        if b.size != self.board_len:
            raise ValueError(f"доска {b.size} элементов, ожидалось {self.board_len}")
        if self._packed:
            row, bad = pack_boards(b.reshape(1, -1))
            if bad.size:
                self._raw[self.rows] = b.reshape(-1).copy()
            self._f.write(row.tobytes())
        else:
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
        self.pos["move"].append(-1 if move is None else move)
        self.pos["move_raw"].append(-1 if move_raw is None else move_raw)

    def close(self):
        """Дописать meta и сделать файлы видимыми. Пустая итерация не
        оставляет за собой ничего."""
        if self._closed:
            return 0
        self._closed = True
        if self in _OPEN:
            _OPEN.remove(self)
        self._f.close()
        # Сигнал может прийти между записью доски и добавлением её столбцов:
        # тогда строк в файле на одну больше, чем в колонках. Режем по общему.
        n_cols = min([len(v) for v in self.pos.values()] + [len(self.pol_idx)])
        if n_cols < self.rows:
            os.truncate(self._tmp, n_cols * self._stride)
            self.rows = n_cols
            self._raw = {r: v for r, v in self._raw.items() if r < n_cols}
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
               "board_len": np.asarray([self.board_len], dtype=np.int32),
               "names": np.asarray(self.names or [""], dtype="<U64")}
        for k, dt in POS_COLS.items():
            out[k] = np.asarray(self.pos[k], dtype=dt)
        for k, dt in GAME_COLS.items():
            out["g_" + k] = np.asarray(self.games[k], dtype=dt)
        meta = os.path.join(self.dir, f"meta_{self.file_id:06d}.npz")
        tmp_meta = meta + ".tmp.npz"        # savez сам добавляет .npz, если его нет
        np.savez(tmp_meta, **out)
        # Сначала meta, потом доски: читатель перечисляет куски по boards_*, и
        # пока этого файла нет, незаконченная пара ему не попадётся.
        os.replace(tmp_meta, meta)
        if self._raw:
            rr = sorted(self._raw)
            np.savez(self._boards_path + ".raw.tmp.npz", rows=np.asarray(rr, dtype=np.int64),
                     boards=np.stack([self._raw[r] for r in rr]))
            os.replace(self._boards_path + ".raw.tmp.npz", self._boards_path + ".raw.npz")
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
            # Кусок в процессе перевода в .fb лежит в обоих форматах — берём один.
            if name.endswith(".f16") and os.path.exists(os.path.join(directory, f"boards_{it:06d}.fb")):
                continue
            if not os.path.exists(meta_path):
                continue
            with np.load(meta_path) as z:
                m = {k: z[k] for k in z.files}
            # Куски, записанные до появления столбца, читаются как «неизвестно»,
            # а не ломают чтение всего архива.
            n_rows = m["game"].shape[0]
            for col, dt in POS_COLS.items():
                if col not in m:
                    m[col] = np.full(n_rows, -1, dtype=dt)
            for col, dt in GAME_COLS.items():
                if "g_" + col not in m:
                    m["g_" + col] = np.full(m["g_result"].shape[0], -1, dtype=dt)
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
                    false_resign=None, decisive=None, source=None):
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
        _in("source", source)
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
            local = self._row[sel]
            order = np.argsort(local)          # последовательное чтение с диска
            dest = np.flatnonzero(np.isin(idx, sel))
            for a in range(0, order.size, batch):
                sl = order[a:a + batch]
                out[dest[sl]] = read_board_rows(path, local[sl], self.board_len)
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

    # ── база данных: поиск по позиции ────────────────────────────────────────

    def position_keys(self, mask=None):
        """Ключ позиции для каждой строки: хеш ТОЛЬКО расстановки фигур, без
        истории и без счётчиков. Две записи с одним ключом — это одна и та же
        позиция на доске, пришедшая из разных партий."""
        import hashlib
        idx = np.flatnonzero(mask) if mask is not None else np.arange(
            self.p["game"].shape[0])
        b = self.boards(np.isin(np.arange(self.p["game"].shape[0]), idx))
        piece_planes = 16 * 80          # 8 наших + 8 чужих, текущая позиция
        return idx, np.array([hashlib.blake2b(row[:piece_planes].tobytes(),
                                              digest_size=8).digest()
                              for row in b])

    def book_from(self, mask=None, min_games=2):
        """Дебютная/эндшпильная книга, выведенная из партий.

        Для каждой позиции, встреченной хотя бы `min_games` раз: какие ходы из
        неё делались, сколько раз и с какой разбивкой побед/ничьих/поражений.
        Это не список чужих рекомендаций, а статистика того, что действительно
        игралось и чем кончилось — как эксплорер на шахматных сайтах.

        Всё считается С ТОЧКИ ЗРЕНИЯ СТОРОНЫ, КОТОРАЯ ХОДИТ: «победа» в записи
        хода чёрных означает победу чёрных. Иначе смешивать ходы обоих цветов в
        одной таблице нельзя.

        Возвращает: ключ позиции → {"n", "w","d","l", "side",
        "moves": {ход_в_координатах_доски: {"n","w","d","l"}}}.
        """
        if mask is None:
            mask = np.ones(self.p["game"].shape[0], dtype=bool)
        # Записи перебора исхода не имеют — в статистике они были бы ничьими.
        mask = mask & (self.g["source"][self.p["game"]] != SOURCE_BOOK)
        idx, keys = self.position_keys(mask)
        side = self.p["side"][idx]
        raw = self.p["move_raw"][idx]
        move_idx = self.p["move"][idx]
        res = self.g["result"][self.p["game"][idx]]
        pov = np.where(side == 0, res, -res)        # исход глазами ходящего
        out = {}
        for i, k in enumerate(keys):
            e = out.setdefault(k, {"n": 0, "w": 0, "d": 0, "l": 0,
                                   "side": int(side[i]), "moves": {}})
            kind = "w" if pov[i] > 0 else ("l" if pov[i] < 0 else "d")
            e["n"] += 1
            e[kind] += 1
            mv = int(raw[i])
            if mv < 0:
                # Куски до появления столбца: ход восстанавливается из
                # канонического индекса, кроме превращений.
                mv = idx_to_move(move_idx[i], side[i])
                if mv is None:
                    continue
            m = e["moves"].setdefault(mv, {"n": 0, "w": 0, "d": 0, "l": 0})
            m["n"] += 1
            m[kind] += 1
        return {k: v for k, v in out.items() if v["n"] >= min_games}


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
def archive_moves(directory, moves, result, source, white="?", black="?",
                  engine=None, iteration=None):
    """Записать одну партию по списку ходов UCI. Возвращает число позиций.

    Общий путь для всего, что играет партии, но не считает распределение
    визитов: часы, CLI, GUI, импорт PGN. Политики у таких позиций нет.
    Нелегальный ход обрывает запись: лучше пропустить партию, чем записать
    мусор."""
    import time as _t
    from capablanca_engine import CapablancaEngine
    eng = engine or CapablancaEngine()
    board_len = int(np.asarray(eng.get_board_tensor()).size)
    w = ArchiveWriter(directory,
                      int(_t.time()) % 1000000 if iteration is None else iteration,
                      board_len)
    rows = []
    for uci in moves:
        mv = _uci_to_move(eng, uci)
        if mv is None:
            return 0
        rows.append((np.asarray(eng.get_board_tensor(), dtype=np.float32),
                     eng.side_to_move(), mv))
        eng.make_move_int(mv)
    if not rows:
        return 0
    if result is None:
        result = int(eng.game_result()) if eng.is_game_over() else 0
    if eng.is_game_over():
        term = (TERM_MATE if abs(result) > 0.5
                else DRAW_REASON_TO_TERM.get(eng.draw_reason(), TERM_DRAW_RULE))
    else:
        term = TERM_LIMIT
    g = w.add_game(result=int(result), plies=len(rows), term=term, playthrough=0,
                   resign_ply=-1, source=source,
                   white_id=w.name_id(white), black_id=w.name_id(black))
    for ply, (board, side, mv) in enumerate(rows):
        w.add_position(board, [], [], game=g, ply=ply, side=side, full=False,
                       root_q=0.0, root_d=-1.0, move=-1, move_raw=int(mv))
    return w.close()


def _uci_to_move(eng, uci):
    for m in eng.get_legal_moves_int():
        if move_to_uci(m & ~0b111 | (m & 0b111)) == uci:
            return m
    return None


def idx_to_move(idx, side):
    """Канонический индекс политики → ход в координатах доски.

    Обычный ход кодируется как `откуда * 80 + куда` по доске, повёрнутой к
    ходящему, поэтому обратим: развернуть обратно при side == 1. Превращения
    (индекс ≥ 6400) хранят только вертикали, не поля, и восстановлению не
    подлежат — для них возвращается None.

    Нужно, чтобы назвать ходы в кусках архива, записанных до появления столбца
    `move_raw`: там индекс есть, а координаты доски — нет.
    """
    idx = int(idx)
    if idx < 0 or idx >= 6400:
        return None
    f, t = idx // 80, idx % 80
    if side == 1:                       # развернуть канонический поворот
        f = (7 - f // 10) * 10 + f % 10
        t = (7 - t // 10) * 10 + t % 10
    return (f << 10) | (t << 3)


PROMO_CHARS = [None, None, 'n', 'b', 'r', 'q', 'a', 'c']


def move_to_uci(m):
    """Ход в координатах доски → строка вида e2e4 / f7f8q. Доска 10 клеток в
    ширину, поэтому поле = ряд*10 + вертикаль."""
    if m is None or m < 0:
        return "?"
    promo = m & 0b111
    t = (m >> 3) & 0x7F
    f = (m >> 10) & 0x7F
    s = (f"{chr(ord('a') + f % 10)}{f // 10 + 1}"
         f"{chr(ord('a') + t % 10)}{t // 10 + 1}")
    # Код превращения = индекс фигуры ПЛЮС ОДИН (2 конь … 7 канцлер), поэтому
    # строка начинается с двух пустых мест. Сдвинутая на единицу таблица
    # называла коня слоном, а канцлера теряла молча.
    return s + PROMO_CHARS[promo] if 0 < promo < len(PROMO_CHARS) else s
