
// src/lib.rs — Capablanca Chess Engine (10x8 board)
use pyo3::prelude::*;

type BB = u128;
const BOARD_MASK: BB = (1u128 << 80) - 1;

fn file_mask(f: u32) -> BB {
    let mut m: BB = 0;
    for r in 0..8u32 { m |= 1u128 << (r * 10 + f); }
    m
}

fn rank_mask(r: u32) -> BB { ((1u128 << 10) - 1) << (r * 10) }
fn not_file_a() -> BB { !file_mask(0) & BOARD_MASK }
fn not_file_b() -> BB { !file_mask(1) & BOARD_MASK }
fn not_file_i() -> BB { !file_mask(8) & BOARD_MASK }
fn not_file_j() -> BB { !file_mask(9) & BOARD_MASK }

fn knight_attacks(sq: u32) -> BB {
    let b: BB = 1u128 << sq;
    let mut m: BB = 0;
    m |= (b << 21) & not_file_a();
    m |= (b << 19) & not_file_j();
    m |= (b >> 19) & not_file_a();
    m |= (b >> 21) & not_file_j();
    m |= (b << 12) & not_file_a() & not_file_b();
    m |= (b << 8)  & not_file_i() & not_file_j();
    m |= (b >> 8)  & not_file_a() & not_file_b();
    m |= (b >> 12) & not_file_i() & not_file_j();
    m & BOARD_MASK
}

fn ray_attacks(sq: u32, occupancy: BB, delta: i32) -> BB {
    let mut attacks: BB = 0;
    let mut current = sq as i32 + delta;
    loop {
        if current < 0 || current >= 80 { break; }
        let curr_file = (current % 10) as u32;
        if delta == 1 || delta == -1 {
            if delta == 1 && curr_file == 0 { break; }
            if delta == -1 && curr_file == 9 { break; }
        }
        if (delta == 11 || delta == -9) && curr_file == 0 { break; }
        if (delta == 9 || delta == -11) && curr_file == 9 { break; }
        let sq_bb: BB = 1u128 << current;
        attacks |= sq_bb;
        if occupancy & sq_bb != 0 { break; }
        current += delta;
    }
    attacks
}

fn bishop_attacks(sq: u32, occ: BB) -> BB { ray_attacks(sq, occ, 11) | ray_attacks(sq, occ, 9) | ray_attacks(sq, occ, -9) | ray_attacks(sq, occ, -11) }
fn rook_attacks(sq: u32, occ: BB) -> BB { ray_attacks(sq, occ, 10) | ray_attacks(sq, occ, -10) | ray_attacks(sq, occ, 1) | ray_attacks(sq, occ, -1) }
fn queen_attacks(sq: u32, occ: BB) -> BB { bishop_attacks(sq, occ) | rook_attacks(sq, occ) }
fn archbishop_attacks(sq: u32, occ: BB) -> BB { bishop_attacks(sq, occ) | knight_attacks(sq) }
fn chancellor_attacks(sq: u32, occ: BB) -> BB { rook_attacks(sq, occ) | knight_attacks(sq) }

fn king_attacks(sq: u32) -> BB {
    let b: BB = 1u128 << sq;
    let not_a = not_file_a();
    let not_j = not_file_j();
    ((b << 10) | (b >> 10) | ((b << 1) & not_a) | ((b >> 1) & not_j) | ((b << 11) & not_a) | ((b << 9) & not_j) | ((b >> 9) & not_a) | ((b >> 11) & not_j)) & BOARD_MASK
}

fn white_pawn_attacks(pawns: BB) -> BB { ((pawns << 11) & not_file_a()) | ((pawns << 9) & not_file_j()) }
fn black_pawn_attacks(pawns: BB) -> BB { ((pawns >> 9) & not_file_a()) | ((pawns >> 11) & not_file_j()) }

const PAWN: usize = 0; const KNIGHT: usize = 1; const BISHOP: usize = 2; const ROOK: usize = 3;
const QUEEN: usize = 4; const ARCH: usize = 5; const CHANC: usize = 6; const KING: usize = 7;

#[derive(Clone)]
pub struct Board {
    pub pieces: [[BB; 8]; 2],
    pub side: usize,
    pub castling: u8,
    pub ep_square: Option<u8>,
    pub halfmove_clock: u32,
    pub fullmove: u32,
}

impl Board {
    fn all_pieces(&self, color: usize) -> BB { self.pieces[color].iter().fold(0, |a, &b| a | b) }
    fn occupancy(&self) -> BB { self.all_pieces(0) | self.all_pieces(1) }

    fn start() -> Self {
        let mut b = Board { pieces: [[0; 8]; 2], side: 0, castling: 0b1111, ep_square: None, halfmove_clock: 0, fullmove: 1 };
        let white_back = [(ROOK, 0), (KNIGHT, 1), (ARCH, 2), (BISHOP, 3), (QUEEN, 4), (KING, 5), (BISHOP, 6), (CHANC, 7), (KNIGHT, 8), (ROOK, 9)];
        for (pt, f) in white_back { b.pieces[0][pt] |= 1u128 << f; }
        b.pieces[0][PAWN] = rank_mask(1);
        let black_back = [(ROOK, 0), (KNIGHT, 1), (ARCH, 2), (BISHOP, 3), (QUEEN, 4), (KING, 5), (BISHOP, 6), (CHANC, 7), (KNIGHT, 8), (ROOK, 9)];
        for (pt, f) in black_back { b.pieces[1][pt] |= 1u128 << (70 + f); }
        b.pieces[1][PAWN] = rank_mask(6);
        b
    }

    fn attacks_by(&self, color: usize) -> BB {
        let occ = self.occupancy();
        let mut att: BB = 0;
        if color == 0 { att |= white_pawn_attacks(self.pieces[0][PAWN]); }
        else { att |= black_pawn_attacks(self.pieces[1][PAWN]); }
        for sq in bb_iter(self.pieces[color][KNIGHT] | self.pieces[color][ARCH] | self.pieces[color][CHANC]) { att |= knight_attacks(sq); }
        for sq in bb_iter(self.pieces[color][BISHOP] | self.pieces[color][ARCH] | self.pieces[color][QUEEN]) { att |= bishop_attacks(sq, occ); }
        for sq in bb_iter(self.pieces[color][ROOK] | self.pieces[color][CHANC] | self.pieces[color][QUEEN]) { att |= rook_attacks(sq, occ); }
        for sq in bb_iter(self.pieces[color][KING]) { att |= king_attacks(sq); }
        att
    }

    fn in_check(&self, color: usize) -> bool { (self.pieces[color][KING] & self.attacks_by(1 - color)) != 0 }

    fn gen_pseudo_legal(&self) -> Vec<(u32, u32, Option<usize>)> {
        let mut moves = Vec::with_capacity(128);
        let us = self.side; let them = 1 - us;
        let occ = self.occupancy();
        let our_pieces = self.all_pieces(us);
        let their_pieces = self.all_pieces(them);
        let empty = !occ & BOARD_MASK;
        let pawns = self.pieces[us][PAWN];

        if us == 0 {
            let push1 = (pawns << 10) & empty;
            let push2 = ((pawns & rank_mask(1)) << 10 & empty) << 10 & empty;
            let cap_r = (pawns << 11) & not_file_a() & their_pieces;
            let cap_l = (pawns << 9) & not_file_j() & their_pieces;
            for to in bb_iter(push1) { add_pawn_move(to - 10, to, us, &mut moves); }
            for to in bb_iter(push2) { moves.push((to - 20, to, None)); }
            // FIX: split loops — объединение через OR теряет ходы когда две пешки бьют одно поле
            for to in bb_iter(cap_r) { add_pawn_move(to - 11, to, us, &mut moves); }
            for to in bb_iter(cap_l) { add_pawn_move(to - 9, to, us, &mut moves); }
            if let Some(ep) = self.ep_square {
                let attackers = ((1u128 << ep >> 11) & not_file_j() | (1u128 << ep >> 9) & not_file_a()) & pawns;
                for from in bb_iter(attackers) { moves.push((from, ep as u32, None)); }
            }
        } else {
            let push1 = (pawns >> 10) & empty;
            let push2 = ((pawns & rank_mask(6)) >> 10 & empty) >> 10 & empty;
            let cap_r = (pawns >> 9) & not_file_a() & their_pieces;
            let cap_l = (pawns >> 11) & not_file_j() & their_pieces;
            for to in bb_iter(push1) { add_pawn_move(to + 10, to, us, &mut moves); }
            for to in bb_iter(push2) { moves.push((to + 20, to, None)); }
            // FIX: split loops — объединение через OR теряет ходы когда две пешки бьют одно поле
            for to in bb_iter(cap_r) { add_pawn_move(to + 9, to, us, &mut moves); }
            for to in bb_iter(cap_l) { add_pawn_move(to + 11, to, us, &mut moves); }
            if let Some(ep) = self.ep_square {
                let attackers = ((1u128 << ep << 11) & not_file_a() | (1u128 << ep << 9) & not_file_j()) & pawns;
                for from in bb_iter(attackers) { moves.push((from, ep as u32, None)); }
            }
        }

        for from in bb_iter(self.pieces[us][KNIGHT]) { for to in bb_iter(knight_attacks(from) & !our_pieces) { moves.push((from, to, None)); } }
        for from in bb_iter(self.pieces[us][BISHOP]) { for to in bb_iter(bishop_attacks(from, occ) & !our_pieces) { moves.push((from, to, None)); } }
        for from in bb_iter(self.pieces[us][ROOK]) { for to in bb_iter(rook_attacks(from, occ) & !our_pieces) { moves.push((from, to, None)); } }
        for from in bb_iter(self.pieces[us][QUEEN]) { for to in bb_iter(queen_attacks(from, occ) & !our_pieces) { moves.push((from, to, None)); } }
        for from in bb_iter(self.pieces[us][ARCH]) { for to in bb_iter(archbishop_attacks(from, occ) & !our_pieces) { moves.push((from, to, None)); } }
        for from in bb_iter(self.pieces[us][CHANC]) { for to in bb_iter(chancellor_attacks(from, occ) & !our_pieces) { moves.push((from, to, None)); } }
        for from in bb_iter(self.pieces[us][KING]) { for to in bb_iter(king_attacks(from) & !our_pieces) { moves.push((from, to, None)); } }
        self.gen_castling(&mut moves);
        moves
    }

    fn gen_castling(&self, moves: &mut Vec<(u32, u32, Option<usize>)>) {
        let us = self.side; let occ = self.occupancy(); let opp_att = self.attacks_by(1 - us);
        let back_rank = if us == 0 { 0u32 } else { 7u32 };
        let king_sq = back_rank * 10 + 5;
        if self.pieces[us][KING] & (1u128 << king_sq) == 0 || (opp_att & (1u128 << king_sq) != 0) { return; }
        if self.castling & (1 << (us * 2)) != 0 {
            let sq6 = 1u128 << (back_rank * 10 + 6); let sq7 = 1u128 << (back_rank * 10 + 7);
            let sq8 = 1u128 << (back_rank * 10 + 8); // FIX: поле i тоже должно быть пустым (ладья встаёт туда)
            if occ & (sq6 | sq7 | sq8) == 0 && (self.pieces[us][ROOK] & (1u128 << (back_rank * 10 + 9)) != 0) && (opp_att & (sq6 | sq7) == 0) { moves.push((king_sq, back_rank * 10 + 7, None)); }
        }
        if self.castling & (1 << (us * 2 + 1)) != 0 {
            let clear = (1u128 << (back_rank * 10 + 1)) | (1u128 << (back_rank * 10 + 2)) | (1u128 << (back_rank * 10 + 3)) | (1u128 << (back_rank * 10 + 4));
            let pass = (1u128 << (back_rank * 10 + 3)) | (1u128 << (back_rank * 10 + 4));
            if occ & clear == 0 && (self.pieces[us][ROOK] & (1u128 << (back_rank * 10 + 0)) != 0) && (opp_att & pass == 0) { moves.push((king_sq, back_rank * 10 + 3, None)); }
        }
    }

    fn gen_legal(&self) -> Vec<(u32, u32, Option<usize>)> {
        self.gen_pseudo_legal().into_iter().filter(|&(f, t, p)| {
            let mut b = self.clone(); b.apply_move(f, t, p); !b.in_check(self.side)
        }).collect()
    }

    fn apply_move(&mut self, from: u32, to: u32, promo: Option<usize>) {
        let us = self.side; let them = 1 - us;
        let from_bb = 1u128 << from; let to_bb = 1u128 << to;
        // FIX: вычисляем взятие ДО очистки pieces[them], иначе информация уже потеряна
        let is_capture = self.all_pieces(them) & to_bb != 0;
        let mut moving_piece = PAWN;
        for p in 0..8 { if self.pieces[us][p] & from_bb != 0 { moving_piece = p; break; } }
        for p in 0..8 { self.pieces[them][p] &= !to_bb; }
        if moving_piece == PAWN {
            if let Some(ep) = self.ep_square { if to == ep as u32 { self.pieces[them][PAWN] &= !(1u128 << (if us == 0 { to - 10 } else { to + 10 })); } }
        }
        self.pieces[us][moving_piece] &= !from_bb; self.pieces[us][moving_piece] |= to_bb;
        if moving_piece == PAWN {
            let promo_rank = if us == 0 { 7 } else { 0 };
            if to / 10 == promo_rank { self.pieces[us][PAWN] &= !to_bb; self.pieces[us][promo.unwrap_or(QUEEN)] |= to_bb; }
        }
        if moving_piece == KING {
            let back = if us == 0 { 0 } else { 70 };
            if from == back + 5 {
                if to == back + 7 { self.pieces[us][ROOK] &= !(1u128 << (back + 9)); self.pieces[us][ROOK] |= 1u128 << (back + 8); }
                else if to == back + 3 { self.pieces[us][ROOK] &= !(1u128 << back); self.pieces[us][ROOK] |= 1u128 << (back + 4); }
            }
            self.castling &= !(3 << (us * 2));
        }
        // FIX: биты рокировки должны совпадать с gen_castling:
        //   бит (us*2)   = королевский фланг → ладья на sq 9 (белые) / 79 (чёрные)
        //   бит (us*2+1) = ферзевый фланг   → ладья на sq 0 (белые) / 70 (чёрные)
        let rook_sqs = [(9u32, 0u8), (0, 1), (79, 2), (70, 3)];
        for (sq, bit) in rook_sqs { if from == sq as u32 || to == sq as u32 { self.castling &= !(1 << bit); } }
        self.ep_square = None;
        if moving_piece == PAWN {
            if us == 0 && from + 20 == to { self.ep_square = Some((from + 10) as u8); }
            else if us == 1 && from == to + 20 { self.ep_square = Some((to + 10) as u8); }
        }
        self.halfmove_clock = if moving_piece == PAWN || is_capture { 0 } else { self.halfmove_clock + 1 };
        if us == 1 { self.fullmove += 1; }
        self.side = them;
    }

    fn material_balance(&self) -> i32 {
        // Стандартные веса + Капабланка-фигуры
        const WEIGHTS: [i32; 8] = [
            1,   // PAWN
            3,   // KNIGHT
            3,   // BISHOP
            5,   // ROOK
            9,   // QUEEN
            8,   // ARCH (Archbishop = Bishop + Knight)
            10,  // CHANC (Chancellor = Rook + Knight)
            0,   // KING
        ];
        let mut score = 0i32;
        for p in 0..8 {
            score += self.pieces[0][p].count_ones() as i32 * WEIGHTS[p];
            score -= self.pieces[1][p].count_ones() as i32 * WEIGHTS[p];
        }
        score
    }

    fn to_tensor(&self) -> Vec<f32> {
        let mut t = vec![0.0f32; 20 * 80];
        for c in 0..2 { for p in 0..8 { for sq in bb_iter(self.pieces[c][p]) { t[(c * 8 + p) * 80 + sq as usize] = 1.0; } } }
        let side_val = if self.side == 0 { 1.0 } else { 0.0 };
        for i in 0..80 { t[16 * 80 + i] = side_val; }
        // FIX: каждый бит рокировки размазан по всей плоскости (все 80 клеток).
        // Старый вариант писал 4 пикселя в угол — CNN не может "видеть" угловые
        // пиксели из середины доски через 3×3 свёртки.
        // Плоскость 17 разбита на 4 зоны по 20 клеток: по одной на каждый бит.
        // Это не требует изменения INPUT_PLANES (остаётся 20).
        for bit in 0u32..4 {
            let val = ((self.castling >> bit) & 1) as f32;
            let base = 17 * 80 + (bit as usize) * 20;
            for i in 0..20 { t[base + i] = val; }
        }
        let hm = (self.halfmove_clock as f32) / 100.0;
        for i in 0..80 { t[18 * 80 + i] = hm; }
        let fm = (self.fullmove as f32) / 200.0;
        for i in 0..80 { t[19 * 80 + i] = fm; }
        t
    }

    fn move_to_idx(from: u32, to: u32, promo: Option<usize>) -> usize {
        match promo {
            None => (from * 80 + to) as usize,
            Some(p) => {
                let pi = match p { QUEEN => 0, ROOK => 1, BISHOP => 2, KNIGHT => 3, ARCH => 4, CHANC => 5, _ => 0 };
                // FIX: включаем file_from чтобы различать две пешки на соседних
                // файлах, которые обе могут пойти/побить на одно поле промоушена.
                // Ранг для промоушенов фиксирован (0 или 7), поэтому (file_from, file_to)
                // однозначно определяет пару (from, to).
                // Диапазон base: 0..=99 (10 файлов × 10 файлов).
                // Макс. индекс: 6400 + 99*6 + 5 = 6999 → POLICY_SIZE = 7000.
                let base = ((from % 10) * 10 + (to % 10)) as usize;
                6400 + base * 6 + pi
            }
        }
    }
}

fn add_pawn_move(from: u32, to: u32, us: usize, moves: &mut Vec<(u32, u32, Option<usize>)>) {
    if to / 10 == (if us == 0 { 7 } else { 0 }) {
        for p in [QUEEN, ROOK, BISHOP, KNIGHT, ARCH, CHANC] { moves.push((from, to, Some(p))); }
    } else { moves.push((from, to, None)); }
}

fn bb_iter(mut bb: BB) -> impl Iterator<Item = u32> {
    std::iter::from_fn(move || { if bb == 0 { None } else { let sq = bb.trailing_zeros(); bb &= bb - 1; Some(sq) } })
}

// FIX: кэшируем легальные ходы — при одной позиции они вычисляются один раз.
// В generate_games каждый ход вызывал gen_legal() трижды:
//   is_game_over() → gen_legal()
//   get_legal_moves_int() → gen_legal()
//   (внутри MCTS copy + expand) → gen_legal()
// Кэш сбрасывается только при make_move_int().
#[pyclass]
#[derive(Clone)]
pub struct CapablancaEngine {
    board: Board,
    legal_cache: Option<Vec<(u32, u32, Option<usize>)>>,
}

impl CapablancaEngine {
    fn ensure_legal_cache(&mut self) {
        if self.legal_cache.is_none() {
            self.legal_cache = Some(self.board.gen_legal());
        }
    }
}

#[pymethods]
impl CapablancaEngine {
    #[new] pub fn new() -> Self { CapablancaEngine { board: Board::start(), legal_cache: None } }
    pub fn copy(&self) -> Self { self.clone() }
    pub fn side_to_move(&self) -> usize { self.board.side }
    pub fn get_board_tensor(&self) -> Vec<f32> { self.board.to_tensor() }

    pub fn get_legal_moves_int(&mut self) -> Vec<u32> {
        self.ensure_legal_cache();
        self.legal_cache.as_ref().unwrap().iter().map(|&(f, t, p)| {
            let p_val = match p { None => 0, Some(pr) => pr as u32 + 1 };
            (f << 10) | (t << 3) | p_val
        }).collect()
    }

    pub fn make_move_int(&mut self, m: u32) -> bool {
        let p_val = m & 0b111;
        let t = (m >> 3) & 0x7F;
        let f = (m >> 10) & 0x7F;
        let p = if p_val == 0 { None } else { Some((p_val - 1) as usize) };
        self.board.apply_move(f, t, p);
        self.legal_cache = None; // сброс кэша после хода
        true
    }

    pub fn move_int_to_policy_idx(&self, m: u32) -> Option<usize> {
        let p_val = m & 0b111;
        let t = (m >> 3) & 0x7F;
        let f = (m >> 10) & 0x7F;
        let p = if p_val == 0 { None } else { Some((p_val - 1) as usize) };
        Some(Board::move_to_idx(f, t, p))
    }

    pub fn is_game_over(&mut self) -> bool {
        if self.board.halfmove_clock >= 100 { return true; }
        self.ensure_legal_cache();
        self.legal_cache.as_ref().unwrap().is_empty()
    }

    pub fn game_result(&mut self) -> f32 {
        if self.board.halfmove_clock >= 100 { return 0.0; }
        self.ensure_legal_cache();
        if self.legal_cache.as_ref().unwrap().is_empty() {
            if self.board.in_check(self.board.side) { return if self.board.side == 0 { -1.0 } else { 1.0 }; }
            return 0.0;
        }
        0.0
    }

    /// Оценка по материалу для позиций где игра прервана по лимиту ходов.
    /// Возвращает: +0.5 (белые лучше), -0.5 (чёрные лучше), 0.0 (равно).
    /// Порог 3 очка — меньше шума от случайных разменов в начале обучения.
    pub fn material_result(&self) -> f32 {
        let balance = self.board.material_balance();
        if balance > 3 { 0.5 }
        else if balance < -3 { -0.5 }
        else { 0.0 }
    }
}

// РЕГИСТРАЦИЯ МОДУЛЯ С ЯВНЫМ ИМЕНЕМ
#[pymodule(name = "capablanca_engine")]
fn capablanca_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CapablancaEngine>()?;
    Ok(())
}
