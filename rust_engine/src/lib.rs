// src/lib.rs — Capablanca Chess Engine (10x8 board)
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray1, IntoPyArray, PyUntypedArrayMethods};
use ndarray::Array2;

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

// Предвычисленные таблицы атак — вычисляются один раз при старте.
// Ускоряет gen_pseudo_legal в 2-3x: не пересчитываем маски на каждый вызов.
use std::sync::OnceLock;

static KNIGHT_ATTACKS: OnceLock<[BB; 80]> = OnceLock::new();
static KING_ATTACKS:   OnceLock<[BB; 80]> = OnceLock::new();

fn init_attack_tables() -> ([BB; 80], [BB; 80]) {
    let mut knights = [0u128; 80];
    let mut kings   = [0u128; 80];
    for sq in 0u32..80 {
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
        knights[sq as usize] = m & BOARD_MASK;

        let not_a = not_file_a(); let not_j = not_file_j();
        let mut k: BB = 0;
        k |= b << 10; k |= b >> 10;
        k |= (b << 1) & not_a; k |= (b >> 1) & not_j;
        k |= (b << 11) & not_a; k |= (b << 9) & not_j;
        k |= (b >> 9)  & not_a; k |= (b >> 11) & not_j;
        kings[sq as usize] = k & BOARD_MASK;
    }
    (knights, kings)
}

fn knight_attacks(sq: u32) -> BB {
    let tables = KNIGHT_ATTACKS.get_or_init(|| init_attack_tables().0);
    tables[sq as usize]
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
    let tables = KING_ATTACKS.get_or_init(|| init_attack_tables().1);
    tables[sq as usize]
}

fn white_pawn_attacks(pawns: BB) -> BB { ((pawns & not_file_j()) << 11) | ((pawns & not_file_a()) << 9) } // FIX: pre-shift маски
fn black_pawn_attacks(pawns: BB) -> BB { ((pawns & not_file_j()) >> 9) | ((pawns & not_file_a()) >> 11) } // FIX: pre-shift маски

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
            let cap_r = ((pawns & not_file_j()) << 11) & their_pieces; // FIX: pre-shift маска
            let cap_l = ((pawns & not_file_a()) << 9)  & their_pieces; // FIX: pre-shift маска
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
            let cap_r = ((pawns & not_file_j()) >> 9)  & their_pieces; // FIX: pre-shift маска
            let cap_l = ((pawns & not_file_a()) >> 11) & their_pieces; // FIX: pre-shift маска
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
        // Рокировка в шахматах Капабланки (10×8):
        //   Королевский фланг: король f(5)→i(8) +3 клетки, ладья j(9)→h(7)
        //     - пусты: g(6), h(7), i(8)
        //     - король не проходит через шах: g(6), h(7), i(8)
        //   Ферзевый фланг: король f(5)→c(2) -3 клетки, ладья a(0)→d(3)
        //     - пусты: b(1), c(2), d(3), e(4)
        //     - король не проходит через шах: e(4), d(3), c(2)
        let us = self.side; let occ = self.occupancy(); let opp_att = self.attacks_by(1 - us);
        let back_rank = if us == 0 { 0u32 } else { 7u32 };
        let b = back_rank * 10;
        let king_sq = b + 5;
        if self.pieces[us][KING] & (1u128 << king_sq) == 0 { return; }
        if opp_att & (1u128 << king_sq) != 0 { return; }

        // ── Королевский фланг: король f(5)→i(8), ладья j(9)→h(7) ──────────
        if self.castling & (1 << (us * 2)) != 0 {
            let must_empty = (1u128 << (b+6)) | (1u128 << (b+7)) | (1u128 << (b+8));
            let king_path  = (1u128 << (b+6)) | (1u128 << (b+7)) | (1u128 << (b+8));
            let rook_ok = self.pieces[us][ROOK] & (1u128 << (b+9)) != 0;
            if occ & must_empty == 0 && rook_ok && opp_att & king_path == 0 {
                moves.push((king_sq, b + 8, None));
            }
        }

        // ── Ферзевый фланг: король f(5)→c(2), ладья a(0)→d(3) ─────────────
        if self.castling & (1 << (us * 2 + 1)) != 0 {
            let must_empty = (1u128 << (b+1)) | (1u128 << (b+2)) | (1u128 << (b+3)) | (1u128 << (b+4));
            let king_path  = (1u128 << (b+4)) | (1u128 << (b+3)) | (1u128 << (b+2));
            let rook_ok = self.pieces[us][ROOK] & (1u128 << (b+0)) != 0;
            if occ & must_empty == 0 && rook_ok && opp_att & king_path == 0 {
                moves.push((king_sq, b + 2, None));
            }
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
            let back = if us == 0 { 0u32 } else { 70u32 };
            if from == back + 5 {
                // Королевский фланг: король f(5)→i(8), ладья j(9)→h(7)
                if to == back + 8 {
                    self.pieces[us][ROOK] &= !(1u128 << (back + 9));
                    self.pieces[us][ROOK] |=   1u128 << (back + 7);
                }
                // Ферзевый фланг: король f(5)→c(2), ладья a(0)→d(3)
                else if to == back + 2 {
                    self.pieces[us][ROOK] &= !(1u128 << (back + 0));
                    self.pieces[us][ROOK] |=   1u128 << (back + 3);
                }
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

    /// Проверяет недостаточность материала для мата.
    ///
    /// В шахматах Капабланки ничья по материалу когда ни одна сторона
    /// не может поставить мат даже при худшей игре противника.
    ///
    /// Фигуры которые ВСЕГДА могут поставить мат (не ничья):
    ///   Пешка, Ладья, Ферзь, Архиепископ (Слон+Конь), Канцлер (Ладья+Конь)
    ///
    /// Случаи недостаточного материала:
    ///   К vs К
    ///   К+Конь vs К
    ///   К+Слон vs К
    ///   К+Конь vs К+Конь
    ///   К+Конь vs К+Слон
    ///   К+Слон vs К+Слон  (любые цвета — при одном слоне у каждого)
    ///   К+Конь+Конь vs К  (два коня не могут форсировать мат)
    ///
    /// НЕ ничья (мат возможен):
    ///   К+Архиепископ vs К  — арх комбинирует ходы коня и слона, мат реален
    ///   К+Канцлер vs К      — тривиальный мат
    ///   К+Ладья vs К        — тривиальный мат
    ///   К+Ферзь vs К        — тривиальный мат
    ///   Любая пешка         — может превратиться
    fn is_insufficient_material(&self) -> bool {
        for c in 0..2 {
            // Если есть пешки, ладьи, ферзи, архиепископы или канцлеры — мат возможен
            if self.pieces[c][PAWN]  != 0 { return false; }
            if self.pieces[c][ROOK]  != 0 { return false; }
            if self.pieces[c][QUEEN] != 0 { return false; }
            if self.pieces[c][ARCH]  != 0 { return false; } // Архиепископ может матовать
            if self.pieces[c][CHANC] != 0 { return false; } // Канцлер может матовать
        }

        // Только короли + лёгкие фигуры (слоны и кони)
        let w_knights = self.pieces[0][KNIGHT].count_ones();
        let w_bishops = self.pieces[0][BISHOP].count_ones();
        let b_knights = self.pieces[1][KNIGHT].count_ones();
        let b_bishops = self.pieces[1][BISHOP].count_ones();
        let w_minor = w_knights + w_bishops;
        let b_minor = b_knights + b_bishops;

        match (w_minor, b_minor) {
            // К vs К
            (0, 0) => true,
            // К+1 vs К  или  К vs К+1
            (1, 0) | (0, 1) => true,
            // К+Конь+Конь vs К — два коня без помощника мат не форсируют
            (2, 0) if w_knights == 2 => true,
            (0, 2) if b_knights == 2 => true,
            // К+1 vs К+1 — слоны и кони не матуют друг против друга
            (1, 1) => true,
            // Всё остальное — мат теоретически возможен
            _ => false,
        }
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

    /// Преобразует ход в индекс policy-вектора в КАНОНИЧЕСКИХ координатах.
    /// При side=1 from/to флипаются вертикально перед кодированием.
    /// Это гарантирует, что policy сети использует одну систему координат
    /// независимо от стороны.
    fn move_to_idx(from: u32, to: u32, promo: Option<usize>, side: usize) -> usize {
        let (f, t) = if side == 1 { (flip_sq(from), flip_sq(to)) } else { (from, to) };
        match promo {
            None => (f * 80 + t) as usize,
            Some(p) => {
                let pi = match p { QUEEN => 0, ROOK => 1, BISHOP => 2, KNIGHT => 3, ARCH => 4, CHANC => 5, _ => 0 };
                // Включаем file_from чтобы различать две пешки на соседних файлах,
                // которые обе могут пойти/побить на одно поле промоушена.
                // base: 0..=99 (10 × 10 файлов).
                // Макс. индекс: 6400 + 99*6 + 5 = 6999 → POLICY_SIZE = 7000.
                let base = ((f % 10) * 10 + (t % 10)) as usize;
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

/// Вертикальный флип квадрата: rank r → rank (7-r), file сохраняется.
/// Для канонической формы инпута: чёрные ходы превращаются в позицию "как если бы белые ходили",
/// чтобы сеть всегда видела доску от лица того, кто ходит (LC0 board.Mirror()).
#[inline]
fn flip_sq(sq: u32) -> u32 { (7 - sq / 10) * 10 + (sq % 10) }

/// Флип всех битов в bitboard.
fn flip_bb(bb: BB) -> BB {
    let mut out: BB = 0;
    let mut b = bb;
    while b != 0 {
        let sq = b.trailing_zeros();
        b &= b - 1;
        out |= 1u128 << flip_sq(sq);
    }
    out
}

// ─── HISTORY PLANES (LC0 encoder.cc) ────────────────────────────────────────
// Сеть видит не только текущую позицию, но и 7 предыдущих → 8 позиций всего.
// Это даёт критический сигнал: "что только что двинулось" (атакующая фигура,
// под боем ли защита), повторения видны напрямую (без специального плана),
// общая динамика позиции.
//
// Plane layout (всего 139 = 8*17 + 3):
//   per history slot h ∈ 0..8 (newest=0):
//     h*17 + 0..7   OUR pieces (P,N,B,R,Q,A,C,K) [canonical-flipped if side=1]
//     h*17 + 8..15  THEIR pieces
//     h*17 + 16     repetition flag (1.0 если эта позиция повторение)
//   136  castling (4 зоны × 20 клеток) — только для текущей позиции
//   137  halfmove / 100
//   138  all-ones (LC0 edge-detection)
pub const HISTORY_LEN: usize = 8;
pub const PLANES_PER_BOARD: usize = 17;  // 8 our + 8 their + 1 rep
pub const META_PLANES: usize = 3;        // castling + halfmove + ones
pub const TOTAL_INPUT_PLANES: usize = HISTORY_LEN * PLANES_PER_BOARD + META_PLANES;  // 139

/// Encode историю позиций в каноническую плоскость для NN.
/// history[0] = текущая позиция (наиболее свежая); history[i] = i ходов назад.
/// Пустые слоты (если истории мало) — заполняются нулями.
/// `current_side` определяет канонический флип ВСЕХ позиций (флипаем относительно
/// стороны, ходящей в ТЕКУЩЕЙ позиции, чтобы "наши" всегда внизу).
/// `rep_flags[i]` = это ли повторение для позиции i.
fn boards_to_tensor(history: &[Board], rep_flags: &[bool],
                    current_side: usize, halfmove: u32, castling: u8) -> Vec<f32> {
    let mut t = vec![0.0f32; TOTAL_INPUT_PLANES * 80];
    let do_flip = current_side == 1;

    for h in 0..HISTORY_LEN {
        if h >= history.len() { break; }
        let board = &history[h];
        // Канонический mapping наших/чужих: "наши" в плоскости — это сторона,
        // которая ходит в ТЕКУЩЕЙ (новейшей) позиции. Для исторических позиций
        // (где могло быть другое состояние) используем то же определение.
        let us = current_side;
        let them = 1 - us;
        let base = h * PLANES_PER_BOARD;
        for p in 0..8 {
            let our_bb = if do_flip { flip_bb(board.pieces[us][p]) } else { board.pieces[us][p] };
            let their_bb = if do_flip { flip_bb(board.pieces[them][p]) } else { board.pieces[them][p] };
            for sq in bb_iter(our_bb)   { t[(base + p) * 80 + sq as usize] = 1.0; }
            for sq in bb_iter(their_bb) { t[(base + 8 + p) * 80 + sq as usize] = 1.0; }
        }
        if rep_flags.get(h).copied().unwrap_or(false) {
            for i in 0..80 { t[(base + 16) * 80 + i] = 1.0; }
        }
    }

    // Meta: castling, halfmove, all-ones — только для текущей позиции
    let us = current_side;
    let bit_order: [u32; 4] = if us == 0 { [0, 1, 2, 3] } else { [2, 3, 0, 1] };
    for (zone, &bit) in bit_order.iter().enumerate() {
        let val = ((castling >> bit) & 1) as f32;
        let base = (HISTORY_LEN * PLANES_PER_BOARD) * 80 + zone * 20;
        for i in 0..20 { t[base + i] = val; }
    }
    let hm = (halfmove as f32 / 100.0).min(1.0);
    let hm_base = (HISTORY_LEN * PLANES_PER_BOARD + 1) * 80;
    for i in 0..80 { t[hm_base + i] = hm; }
    let ones_base = (HISTORY_LEN * PLANES_PER_BOARD + 2) * 80;
    for i in 0..80 { t[ones_base + i] = 1.0; }
    t
}

// FIX: кэшируем легальные ходы — при одной позиции они вычисляются один раз.
// В generate_games каждый ход вызывал gen_legal() трижды:
//   is_game_over() → gen_legal()
//   get_legal_moves_int() → gen_legal()
//   (внутри MCTS copy + expand) → gen_legal()
// Кэш сбрасывается только при make_move_int().
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct CapablancaEngine {
    board: Board,
    legal_cache: Option<Vec<(u32, u32, Option<usize>)>>,
    position_history: Vec<u64>,
    // board_history: snapshot последних HISTORY_LEN досок ВКЛЮЧАЯ текущую.
    // board_history[0] = текущая позиция, board_history[1] = ход назад, и т.д.
    // Используется для history planes (LC0-style).
    board_history: Vec<Board>,
}

impl CapablancaEngine {
    fn ensure_legal_cache(&mut self) {
        if self.legal_cache.is_none() {
            self.legal_cache = Some(self.board.gen_legal());
        }
    }

    /// Возвращает true если позиция в board_history[i] — повторение
    /// (хеш встречается ещё хотя бы раз в position_history до этой точки).
    /// Для history planes — даёт сети сигнал о повторениях по слотам.
    fn rep_flags(&self) -> Vec<bool> {
        self.board_history.iter().map(|b| {
            let h = compute_board_hash(b);
            // Сколько раз эта позиция встречалась всего в game_history.
            // Если > 1 — повторение. (Текущая позиция тоже в position_history.)
            let count = self.position_history.iter().filter(|&&x| x == h).count();
            count > 1
        }).collect()
    }
}

#[pymethods]
impl CapablancaEngine {
    #[new] pub fn new() -> Self {
        let board = Board::start();
        let h = compute_board_hash(&board);
        CapablancaEngine {
            board: board.clone(),
            legal_cache: None,
            position_history: vec![h],
            board_history: vec![board],
        }
    }
    pub fn copy(&self) -> Self { self.clone() }
    pub fn side_to_move(&self) -> usize { self.board.side }

    /// Список фигур на доске в RAW (non-canonical) координатах.
    /// Возвращает Vec<(color, piece_type, square)>:
    ///   color: 0=белые, 1=чёрные
    ///   piece_type: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, ARCH=5, CHANC=6, KING=7
    ///   square: 0..80, square = rank*10 + file
    /// Используется GUI для отрисовки — там нужен RAW вид, не каноническая флипнутая форма.
    pub fn get_pieces(&self) -> Vec<(usize, usize, u32)> {
        let mut out = Vec::with_capacity(40);
        for color in 0..2usize {
            for piece in 0..8usize {
                for sq in bb_iter(self.board.pieces[color][piece]) {
                    out.push((color, piece, sq));
                }
            }
        }
        out
    }

    pub fn get_board_tensor(&self) -> Vec<f32> {
        // History planes (LC0-style): сеть видит 8 последних позиций (newest first).
        // Текущая = board_history[0], предыдущая = board_history[1], и т.д.
        let reps = self.rep_flags();
        boards_to_tensor(
            &self.board_history,
            &reps,
            self.board.side,
            self.board.halfmove_clock,
            self.board.castling,
        )
    }

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
        // Поддерживаем историю позиций для 3-fold repetition.
        // При необратимом ходе (взятие/пешка) halfmove_clock=0 → старые позиции не повторятся.
        if self.board.halfmove_clock == 0 {
            self.position_history.clear();
        }
        self.position_history.push(compute_board_hash(&self.board));
        // board_history: pushуем НОВУЮ текущую позицию в front, обрезаем до HISTORY_LEN.
        // board_history[0] = текущая, [1] = предыдущая, ...
        self.board_history.insert(0, self.board.clone());
        if self.board_history.len() > HISTORY_LEN {
            self.board_history.truncate(HISTORY_LEN);
        }
        true
    }

    pub fn move_int_to_policy_idx(&self, m: u32) -> Option<usize> {
        let p_val = m & 0b111;
        let t = (m >> 3) & 0x7F;
        let f = (m >> 10) & 0x7F;
        let p = if p_val == 0 { None } else { Some((p_val - 1) as usize) };
        // Канонический индекс: при чёрном ходе from/to флипаются.
        Some(Board::move_to_idx(f, t, p, self.board.side))
    }

    pub fn is_game_over(&mut self) -> bool {
        if self.board.halfmove_clock >= 100 { return true; }
        if self.board.is_insufficient_material() { return true; }
        // 3-fold: текущая позиция уже в history (push в make_move_int + new()).
        let cur = compute_board_hash(&self.board);
        let repeats = self.position_history.iter().filter(|&&h| h == cur).count();
        if repeats >= 3 { return true; }
        self.ensure_legal_cache();
        self.legal_cache.as_ref().unwrap().is_empty()
    }

    pub fn game_result(&mut self) -> f32 {
        if self.board.halfmove_clock >= 100 { return 0.0; }
        if self.board.is_insufficient_material() { return 0.0; }
        // 3-fold repetition = ничья
        let cur = compute_board_hash(&self.board);
        let repeats = self.position_history.iter().filter(|&&h| h == cur).count();
        if repeats >= 3 { return 0.0; }
        self.ensure_legal_cache();
        if self.legal_cache.as_ref().unwrap().is_empty() {
            if self.board.in_check(self.board.side) {
                return if self.board.side == 0 { -1.0 } else { 1.0 };
            }
            return 0.0; // Пат
        }
        0.0
    }

    /// Досрочное присуждение результата по материалу (Adjudication).
    ///
    /// Возвращает:
    ///   ±1.0 — решительный перевес (≥8 очков): одна сторона имеет крупную фигуру
    ///          которой у противника нет. Засчитывается как победа.
    ///   ±0.5 — умеренный перевес (3-7 очков): вероятная победа, но не гарантия.
    ///    0.0 — примерное равенство (< 3 очков).
    ///
    /// Порог 8 очков выбран под Капабланку: архиепископ стоит ~8, канцлер ~10.
    /// Если у одной стороны есть арх/канц/ферзь, а у другой нет — перевес решающий.
    pub fn material_result(&self) -> f32 {
        let balance = self.board.material_balance();
        if balance >= 8  { return  1.0; }
        if balance <= -8 { return -1.0; }
        if balance > 3  { return  0.5; }
        if balance < -3 { return -0.5; }
        0.0
    }

    pub fn adjudication_result(&self) -> Option<f32> { None }
}

// РЕГИСТРАЦИЯ МОДУЛЯ С ЯВНЫМ ИМЕНЕМ
#[pymodule(name = "capablanca_engine")]
fn capablanca_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CapablancaEngine>()?;
    m.add_class::<RustMCTS>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_indices_no_collision() {
        let mut seen = std::collections::HashMap::new();
        let _board = Board::start();
        let promos = [QUEEN, ROOK, BISHOP, KNIGHT, ARCH, CHANC];
        for from_file in 0u32..10 {
            for to_file in from_file.saturating_sub(1)..=(from_file+1).min(9) {
                for &p in &promos {
                    let from_sq = 6 * 10 + from_file;
                    let to_sq   = 7 * 10 + to_file;
                    let idx = Board::move_to_idx(from_sq, to_sq, Some(p), 0);
                    assert!(idx < 7000, "idx={idx} >= POLICY_SIZE=7000 for promo {p}");
                    let key = (from_sq, to_sq, p);
                    if let Some(prev) = seen.insert(idx, key) {
                        panic!("Collision at idx={idx}: {:?} vs {:?}", prev, key);
                    }
                }
            }
        }
        for f in 0u32..80 {
            for t in 0u32..80 {
                if f == t { continue; }
                let idx = Board::move_to_idx(f, t, None, 0);
                assert!(idx < 6400, "Normal move idx={idx} >= 6400, from={f} to={t}");
            }
        }
        println!("✅ policy indices: no collisions, all within POLICY_SIZE=7000");
    }

    #[test]
    fn test_canonical_flip_symmetry() {
        // Любой ход за чёрных в канонических координатах должен иметь
        // ТОТ ЖЕ индекс, что симметричный ход за белых.
        // Пример: e2e4 белых = e7e5 чёрных в канонике.
        let promos = [QUEEN, ROOK, BISHOP, KNIGHT, ARCH, CHANC];
        for from_file in 0u32..10 {
            for to_file in 0u32..10 {
                for from_rank in 0u32..8 {
                    for to_rank in 0u32..8 {
                        let from_w = from_rank * 10 + from_file;
                        let to_w   = to_rank   * 10 + to_file;
                        // Симметричный ход за чёрных: тот же ход на флипнутой доске
                        let from_b = (7 - from_rank) * 10 + from_file;
                        let to_b   = (7 - to_rank)   * 10 + to_file;
                        assert_eq!(
                            Board::move_to_idx(from_w, to_w, None, 0),
                            Board::move_to_idx(from_b, to_b, None, 1),
                            "white {from_w}→{to_w} should map to same idx as black {from_b}→{to_b}"
                        );
                        for &p in &promos {
                            assert_eq!(
                                Board::move_to_idx(from_w, to_w, Some(p), 0),
                                Board::move_to_idx(from_b, to_b, Some(p), 1),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_boards_to_tensor_canonical() {
        // Каноническая форма: начальная позиция за белых и чёрных должна совпадать
        // в первых 8 piece-планах (OUR pieces всегда на bottom rank).
        let b_white = Board::start();
        let mut b_black = Board::start();
        b_black.side = 1;
        let t_w = boards_to_tensor(&[b_white.clone()], &[false], 0, 0, 0b1111);
        let t_b = boards_to_tensor(&[b_black.clone()], &[false], 1, 0, 0b1111);
        for plane in 0..8 {
            for sq in 0..80 {
                assert_eq!(t_w[plane * 80 + sq], t_b[plane * 80 + sq],
                    "OUR plane {plane} sq {sq} mismatch w={} b={}",
                    t_w[plane * 80 + sq], t_b[plane * 80 + sq]);
            }
        }
        // Размер тензора = TOTAL_INPUT_PLANES * 80
        assert_eq!(t_w.len(), TOTAL_INPUT_PLANES * 80);
    }

    #[test]
    fn test_history_planes_layout() {
        // 8 history slots + 3 meta = 139 plane (для 17 planes per board)
        assert_eq!(TOTAL_INPUT_PLANES, HISTORY_LEN * PLANES_PER_BOARD + META_PLANES);
        assert_eq!(TOTAL_INPUT_PLANES, 139);
    }

    #[test]
    fn test_root_is_not_repetition() {
        // КРИТИЧЕСКИЙ regression-тест:
        // При первом expand root'а — position_history содержит root_hash (single entry).
        // rep_count_at_leaf(root, root_board) ДОЛЖЕН вернуть 1 (только history),
        // не 2 — иначе root помечается как 2-fold терминал → MCTS не работает.
        let mcts = SingleMcts::new(Board::start());
        let root = mcts.root;
        let count = mcts.rep_count_at_leaf(root, &mcts.root_board);
        assert_eq!(count, 1, "root не должен считаться повторением при первом expand");
        assert!(!mcts.is_repetition_at_leaf(root, &mcts.root_board),
                "root не должен быть 2-fold терминалом");
    }

    #[test]
    fn test_startpos_legal_moves() {
        let mut board = Board::start();
        let legal = board.gen_legal();
        // В начальной позиции Капабланки: 10 пешечных ходов + 4 хода конями + 4 хода архиепископа/канцлера
        // Точное число зависит от правил, но должно быть > 20
        assert!(legal.len() >= 20, "Too few legal moves at startpos: {}", legal.len());
        assert!(legal.len() <= 50, "Too many legal moves at startpos: {}", legal.len());
        println!("Legal moves at start: {}", legal.len());
    }

    #[test]
    fn test_board_hash_unique() {
        let b1 = Board::start();
        let mut b2 = Board::start();
        b2.apply_move(1, 22, None);
        let h1 = SingleMcts::board_hash(&b1);
        let h2 = SingleMcts::board_hash(&b2);
        assert_ne!(h1, h2, "Different positions must have different hashes");
    }

    #[test]
    fn test_insufficient_material() {
        // Вспомогательная функция: создаёт пустую доску только с королями
        fn kings_only() -> Board {
            let mut b = Board { pieces: [[0; 8]; 2], side: 0, castling: 0,
                ep_square: None, halfmove_clock: 0, fullmove: 1 };
                b.pieces[0][KING] = 1u128 << 5;   // белый король f1
                b.pieces[1][KING] = 1u128 << 75;  // чёрный король f8
                b
        }

        // К vs К — ничья
        let b = kings_only();
        assert!(b.is_insufficient_material(), "K vs K must be draw");

        // К+Конь vs К — ничья
        let mut b = kings_only();
        b.pieces[0][KNIGHT] = 1u128 << 1;
        assert!(b.is_insufficient_material(), "K+N vs K must be draw");

        // К+Слон vs К — ничья
        let mut b = kings_only();
        b.pieces[0][BISHOP] = 1u128 << 3;
        assert!(b.is_insufficient_material(), "K+B vs K must be draw");

        // К+2 Коня vs К — ничья (форсированный мат невозможен)
        let mut b = kings_only();
        b.pieces[0][KNIGHT] = (1u128 << 1) | (1u128 << 8);
        assert!(b.is_insufficient_material(), "K+N+N vs K must be draw");

        // К+1 vs К+1 — ничья
        let mut b = kings_only();
        b.pieces[0][KNIGHT] = 1u128 << 1;
        b.pieces[1][BISHOP] = 1u128 << 66;
        assert!(b.is_insufficient_material(), "K+N vs K+B must be draw");

        // К+Архиепископ vs К — НЕ ничья
        let mut b = kings_only();
        b.pieces[0][ARCH] = 1u128 << 2;
        assert!(!b.is_insufficient_material(), "K+A vs K must NOT be draw");

        // К+Канцлер vs К — НЕ ничья
        let mut b = kings_only();
        b.pieces[0][CHANC] = 1u128 << 7;
        assert!(!b.is_insufficient_material(), "K+C vs K must NOT be draw");

        // К+Ладья vs К — НЕ ничья
        let mut b = kings_only();
        b.pieces[0][ROOK] = 1u128 << 0;
        assert!(!b.is_insufficient_material(), "K+R vs K must NOT be draw");

        // К+Пешка vs К — НЕ ничья
        let mut b = kings_only();
        b.pieces[0][PAWN] = 1u128 << 10;
        assert!(!b.is_insufficient_material(), "K+P vs K must NOT be draw");

        println!("✅ insufficient material: all cases correct");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUST MCTS — дерево живёт в Rust, Python только кормит нейросетью
// ═══════════════════════════════════════════════════════════════════════════════

const POLICY_SIZE_MCTS: usize = 7000;
const VIRTUAL_LOSS_V: i32 = 3;
// Параметры MCTS из lc0 params.cc — тюненные на миллионах партий.
const C_PUCT_V: f32 = 1.745;        // CPuct (lc0)
const C_PUCT_FACTOR: f32 = 3.894;   // CPuctFactor — множитель для логарифмического роста
const C_PUCT_BASE: f32 = 38739.0;   // CPuctBase — точка перегиба
// Dirichlet alpha вычисляется динамически: max(10/n_children, 0.1). См. expand().
const DIRICHLET_EPS_V: f64 = 0.35; // повышено с 0.25 — больше исследования на старте

// MLH normalization (должно совпадать с CapablancaNet.MLH_PLY_NORM в Python).
// NN выдаёт [0,1] = "доля от MLH_PLY_NORM полуходов". Умножаем при backup.
const MLH_PLY_NORM: f32 = 200.0;
// LC0 MEvaluator константы (params.cc:597-600). Эффект включается только при |q| > THRESHOLD.
const M_SLOPE: f32 = 0.0027;
const M_CAP: f32 = 0.0345;
const M_THRESHOLD: f32 = 0.8;

// MctsNode без Board — как в lc0.
// Board хранится только в SingleMcts.root_board.
// Позиция восстанавливается при expand/collect_leaves проходом от корня.
// Размер: ~62 байт вместо ~300 байт → 8192 узлов = 500KB (влезает в L2).
//
// Bounds (lower, upper): доказанный диапазон значения узла от его POV.
//   -1 = LOSS, 0 = DRAW, +1 = WIN. По умолчанию (-1, +1) = ничего не доказано.
//   Терминалы: lower == upper == результат игры.
//   Bounds prop. (LC0 StickyEndgames, search.cc:2302) — когда все дети теряют →
//   родитель доказан как выигрыш. Используется в select() для перенаправления
//   симуляций из решённых веток.
// Типы терминалов — для корректного tree reuse.
// Natural терминалы (мат/пат/50-move/insufficient) — ПЕРМАНЕНТНЫЕ (от позиции).
// TwoFold — path-dependent: после shift корня может стать невалидным.
// Default (0) = либо нетерминал, либо bounds-proven (тоже path-dependent через детей).
const TERMINAL_KIND_NONE: u8 = 0;     // не терминал ИЛИ bounds-proven
const TERMINAL_KIND_NATURAL: u8 = 1;  // мат/пат/50-move/insufficient — permanent
const TERMINAL_KIND_TWOFOLD: u8 = 2;  // 2-fold внутри дерева — path-dependent

struct MctsNode {
    move_from_parent: u32,  // ход которым пришли в этот узел
    prior: f32,
    visits: i32,
    wl: f32,               // running average Q (lc0: FinalizeScoreUpdate)
    d: f32,                // running average Draw probability (для contempt: Q = WL + draw_score*D)
    m: f32,                // running average remaining plies (LC0 MLH). В PLY-единицах (не нормализовано).
    virtual_loss: i32,
    children: Vec<usize>,
    is_expanded: bool,
    is_terminal: bool,
    terminal_kind: u8,     // см. TERMINAL_KIND_*
    lower: i8,             // нижняя граница значения (proven worst-case) от POV этого узла
    upper: i8,             // верхняя граница значения (proven best-case)
    side: u8,              // чья очередь хода в этом узле (0=белые, 1=чёрные)
    parent: Option<usize>,
    // Position hash для O(1) repetition lookup. Заполняется в expand().
    // 0 для нераскрытых узлов — невалидно для сравнения.
    position_hash: u64,
}

impl MctsNode {
    fn new(move_from_parent: u32, prior: f32, side: u8, parent: Option<usize>) -> Self {
        MctsNode {
            move_from_parent, prior, side,
            visits: 0, wl: 0.0, d: 0.0, m: 0.0, virtual_loss: 0,
            children: Vec::new(),
            is_expanded: false, is_terminal: false,
            terminal_kind: TERMINAL_KIND_NONE,
            lower: -1, upper: 1,  // ничего не доказано
            parent,
            position_hash: 0,
        }
    }
    fn q(&self) -> f32 {
        if self.virtual_loss > 0 {
            let total = self.visits + self.virtual_loss;
            if total > 0 { (self.wl * self.visits as f32 - self.virtual_loss as f32) / total as f32 }
            else { 0.0 }
        } else {
            self.wl
        }
    }
}

struct Arena { nodes: Vec<MctsNode> }
impl Arena {
    fn new(cap: usize) -> Self { Arena { nodes: Vec::with_capacity(cap) } }
    fn add(&mut self, n: MctsNode) -> usize { let i = self.nodes.len(); self.nodes.push(n); i }
    fn get(&self, i: usize) -> &MctsNode { &self.nodes[i] }
    fn get_mut(&mut self, i: usize) -> &mut MctsNode { &mut self.nodes[i] }
}

fn xorshift64(s: &mut u64) -> f64 {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    (*s as f64) / (u64::MAX as f64)
}

// FIX: оригинальный Marsaglia-Tsang работает только при alpha >= 1/3.
// При alpha=0.3: d = 0.3 - 1/3 = -0.033 → sqrt(9*d) = sqrt(-0.3) = NaN.
// NaN-приоритеты → select() всегда выбирал children[0] → entropy=0, top1=1.
// Исправление: sample_gamma с редукцией Gamma(a) = Gamma(a+1) * U^(1/a) для a < 1.
fn sample_gamma(alpha: f64, rng: &mut u64) -> f64 {
    if alpha >= 1.0 {
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = xorshift64(rng).max(1e-15);
            let u2 = xorshift64(rng);
            let x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 { continue; }
            let u = xorshift64(rng);
            if u < 1.0 - 0.0331 * x.powi(4) { return d * v; }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) { return d * v; }
        }
    } else {
        // Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha), корректно для любого alpha > 0
        let g = sample_gamma(alpha + 1.0, rng);
        let u = xorshift64(rng).max(1e-15);
        g * u.powf(1.0 / alpha)
    }
}

fn dirichlet_noise(alpha: f64, n: usize, rng: &mut u64) -> Vec<f64> {
    let mut out: Vec<f64> = (0..n).map(|_| sample_gamma(alpha, rng)).collect();
    let sum: f64 = out.iter().sum();
    if sum > 0.0 { out.iter_mut().for_each(|x| *x /= sum); }
    out
}

struct SingleMcts {
    arena: Arena,
    root: usize,
    root_board: Board,
    pending: Vec<usize>,
    pending_boards: Vec<Board>,
    position_history: Vec<u64>,
    // root_history[0] = root_board, [1..] = ходы НАЗАД от root (до HISTORY_LEN-1 элементов).
    // При leaf-walk собираем history путём prepend новых позиций.
    root_history: Vec<Board>,
    // Переиспользуемый буфер для collect_leaves — без аллокаций каждый шаг
    leaf_tensor_buf: Vec<f32>,
    // KLD-early-exit: snapshot предыдущего распределения root visits.
    // Vec<(move_int, visits)>. None если snapshot ещё не делался (или после сброса root).
    // Считается в Rust для избежания marshalling 7000-vector в Python каждые 2 шага MCTS.
    kld_prev_snapshot: Option<Vec<(u32, i32)>>,
    // Accumulating tree reuse: visits корня НА МОМЕНТ начала текущего хода
    // (т.е. визиты унаследованные через tree reuse от прошлого хода).
    // sims_done_this_move = root.visits - move_start_visits → честный остаток
    // нового бюджета для best_move_is_decided, без "удушения" из-за reused визитов.
    move_start_visits: i32,
}

fn compute_board_hash(b: &Board) -> u64 {
    let mut h: u64 = b.side as u64 * 0x9e3779b97f4a7c15;
    for c in 0..2usize {
        for p in 0..8usize {
            let lo = b.pieces[c][p] as u64;
            let hi = (b.pieces[c][p] >> 64) as u64;
            h ^= lo.wrapping_mul(0x517cc1b727220a95u64.wrapping_add((c * 8 + p) as u64));
            h ^= hi.wrapping_mul(0xbf58476d1ce4e5b9u64.wrapping_add((c * 8 + p) as u64));
            h = h.rotate_left(17);
        }
    }
    h ^= (b.castling as u64).wrapping_mul(0x6c62272e07bb0142);
    h ^= b.ep_square.map(|s| s as u64 + 1).unwrap_or(0).wrapping_mul(0x94d049bb133111eb);
    h
}

impl SingleMcts {
    fn board_hash(b: &Board) -> u64 { compute_board_hash(b) }

    fn new(board: Board) -> Self {
        Self::new_with_history(board.clone(), vec![board])
    }

    fn new_with_history(board: Board, root_history: Vec<Board>) -> Self {
        let side = board.side as u8;
        let initial_hash = Self::board_hash(&board);
        let mut arena = Arena::new(8192);
        let root = arena.add(MctsNode::new(0, 1.0, side, None));
        arena.get_mut(root).position_hash = initial_hash;  // root всегда имеет валидный хеш
        let buf_cap = 8192 * TOTAL_INPUT_PLANES * 80;
        SingleMcts {
            arena, root, root_board: board,
            pending: Vec::new(), pending_boards: Vec::new(),
            position_history: vec![initial_hash],
            root_history,
            leaf_tensor_buf: Vec::with_capacity(buf_cap),
            kld_prev_snapshot: None,
            move_start_visits: 0,
        }
    }

    /// Снимок текущего распределения root visits для KLD-early-exit.
    /// Сохраняет (move_int, visits) для каждого root-child.
    fn kld_take_snapshot(&mut self) {
        let root_children = self.arena.get(self.root).children.clone();
        let mut snap: Vec<(u32, i32)> = Vec::with_capacity(root_children.len());
        for ci in root_children {
            let c = self.arena.get(ci);
            snap.push((c.move_from_parent, c.visits));
        }
        self.kld_prev_snapshot = Some(snap);
    }

    /// KL(prev || curr) по root visits. Возвращает +inf если snapshot нет
    /// или у одной из сторон 0 визитов (KL не определён).
    /// O(N_children^2) из-за линейного поиска, но N обычно < 60.
    fn kld_compute_gain(&self) -> f32 {
        let prev = match &self.kld_prev_snapshot {
            Some(s) if !s.is_empty() => s,
            _ => return f32::INFINITY,
        };
        let root = self.arena.get(self.root);
        let curr_total: i32 = root.children.iter().map(|&ci| self.arena.get(ci).visits).sum();
        let prev_total: i32 = prev.iter().map(|(_, v)| *v).sum();
        if curr_total <= 0 || prev_total <= 0 { return f32::INFINITY; }
        let pt = prev_total as f32;
        let ct = curr_total as f32;
        let eps = 1e-8_f32;
        let mut kl = 0.0_f32;
        for &(m, pv) in prev {
            if pv == 0 { continue; }
            let p = pv as f32 / pt;
            // Найти curr visits для move m (линейный поиск; N_children обычно < 60)
            let mut cv: i32 = 0;
            for &ci in &root.children {
                if self.arena.get(ci).move_from_parent == m {
                    cv = self.arena.get(ci).visits;
                    break;
                }
            }
            let q = cv as f32 / ct;
            kl += p * ((p + eps).ln() - (q + eps).ln());
        }
        kl
    }

    /// Сбрасывает KLD snapshot. Вызывается при make_move (root shift),
    /// renoise_root или любом изменении корневой структуры.
    fn kld_reset(&mut self) {
        self.kld_prev_snapshot = None;
    }

    // Восстанавливает Board для узла idx, проходя путь от корня.
    // O(depth) — обычно 10-30 ходов, быстро.
    fn board_at(&self, idx: usize) -> Board {
        // Собираем путь от узла до корня
        let mut path = Vec::new();
        let mut cur = idx;
        while cur != self.root {
            let node = self.arena.get(cur);
            path.push(node.move_from_parent);
            match node.parent {
                Some(p) => cur = p,
                None => break,
            }
        }
        // Применяем ходы от корня вниз
        let mut board = self.root_board.clone();
        for &m in path.iter().rev() {
            if m != 0 {
                let pv = m & 0b111;
                let t  = (m >> 3) & 0x7F;
                let f  = (m >> 10) & 0x7F;
                let p  = if pv == 0 { None } else { Some((pv - 1) as usize) };
                board.apply_move(f, t, p);
            }
        }
        board
    }

    /// Восстанавливает Board И history (HISTORY_LEN последних позиций) для узла idx.
    /// history[0] = leaf, history[1] = на 1 ход назад, ..., history[k] = root_history[k - path_len].
    /// Используется для history-planes encoding листа.
    fn board_with_history_at(&self, idx: usize) -> (Board, Vec<Board>) {
        let mut path = Vec::new();
        let mut cur = idx;
        while cur != self.root {
            let node = self.arena.get(cur);
            path.push(node.move_from_parent);
            match node.parent {
                Some(p) => cur = p,
                None => break,
            }
        }
        // Replay forward, фиксируя промежуточные позиции
        let mut board = self.root_board.clone();
        let mut forward_boards: Vec<Board> = Vec::with_capacity(path.len() + 1);
        forward_boards.push(board.clone());  // root
        for &m in path.iter().rev() {
            if m != 0 {
                let pv = m & 0b111;
                let t  = (m >> 3) & 0x7F;
                let f  = (m >> 10) & 0x7F;
                let p  = if pv == 0 { None } else { Some((pv - 1) as usize) };
                board.apply_move(f, t, p);
            }
            forward_boards.push(board.clone());
        }
        let leaf = forward_boards.last().unwrap().clone();

        // Собираем history newest→oldest:
        //   сначала пути в дереве (от leaf к root, не включая root_board дублирующий)
        //   затем root_history[1..] (root_board уже взяли как последний path-элемент)
        let mut history: Vec<Board> = Vec::with_capacity(HISTORY_LEN);
        for b in forward_boards.iter().rev() {
            history.push(b.clone());
            if history.len() >= HISTORY_LEN { break; }
        }
        if history.len() < HISTORY_LEN {
            // root_history[0] == self.root_board → пропускаем, берём с [1]
            for b in self.root_history.iter().skip(1) {
                history.push(b.clone());
                if history.len() >= HISTORY_LEN { break; }
            }
        }
        (leaf, history)
    }

    fn select(&mut self) -> Option<usize> {
        let mut idx = self.root;
        loop {
            // Терминалы: бэкап известного значения вверх (LC0-style).
            // Без этого симуляция теряется → parent Q не сходится к терминалу,
            // PUCT продолжает экспериментировать вместо использования известного исхода.
            let (is_terminal, terminal_v, terminal_d, terminal_m) = {
                let n = self.arena.get(idx);
                (n.is_terminal, n.wl, n.d, n.m)
            };
            if is_terminal {
                // Терминал: m = 0 (игра окончена), plies_from_leaf инкрементируется в backup
                self.backup(idx, terminal_v, terminal_d, terminal_m);
                return None;
            }
            let node = self.arena.get(idx);
            if !node.is_expanded { return Some(idx); }
            if node.children.is_empty() { return None; }

            let parent_visits = (self.arena.get(idx).visits + self.arena.get(idx).virtual_loss).max(1);
            let sqrt_n = (parent_visits as f32).sqrt();

            // Динамический CPUCT по формуле lc0 (params.cc):
            //   cpuct = CPuct + CPuctFactor * ln((N + CPuctBase) / CPuctBase)
            // Тюненные значения: CPuct=1.745, CPuctFactor=3.894, CPuctBase=38739
            // При N=0: cpuct ≈ 1.745, при N=10K: ≈ 2.7, при N=100K: ≈ 4.0
            let cpuct = C_PUCT_V
                + C_PUCT_FACTOR * ((parent_visits as f32 + C_PUCT_BASE) / C_PUCT_BASE).ln();

            // FPU стратегии (LC0 params.cc:567-572):
            //   non-root: "reduction" — fpu = parent_q - 0.330 * sqrt(visited_policy_sum)
            //   root:     "absolute" — fpu = +1.0 (раздать максимум exploration корневым детям)
            // На корне непосещённые ходы получают максимально оптимистичный score → быстро
            // сканируется ВЕСЬ корневой move set прежде чем углубляться в один из них.
            // Это важно для Dirichlet noise: иначе noise работает только когда дети посещены,
            // а на старте все дети непосещены и получают одинаковый fpu=parent_q.
            let parent_q = self.arena.get(idx).q();
            let n_ch = self.arena.get(idx).children.len();
            let is_root = idx == self.root;
            let fpu = if is_root {
                1.0f32  // absolute strategy: forced exploration на корне
            } else {
                const FPU_REDUCTION: f32 = 0.330;
                let mut visited_pol = 0.0f32;
                for ci_pos in 0..n_ch {
                    let ci = self.arena.get(idx).children[ci_pos];
                    let c = self.arena.get(ci);
                    if c.visits > 0 || c.virtual_loss > 0 { visited_pol += c.prior; }
                }
                (parent_q - FPU_REDUCTION * visited_pol.sqrt()).max(-1.0)
            };

            let mut best = f32::NEG_INFINITY;
            let mut best_ci = self.arena.get(idx).children[0];
            // Pre-compute данные для M-utility (LC0 MEvaluator, search.cc:111).
            // Эффект включается только если parent_q "сильно решён" (|q| > THRESHOLD).
            // Это делает M-bias дешёвым в обычных позициях и сильным в эндшпиле.
            let parent_m = self.arena.get(idx).m;
            let m_active = parent_q.abs() > M_THRESHOLD;

            // Early-exit after seeing first unvisited child: children are sorted by prior
            // descending (SortEdges in expand), so the first unvisited is always the
            // best unvisited. After it, one more step finds second-best; then we stop.
            let mut can_exit = false;
            for ci_pos in 0..n_ch {
                let ci = self.arena.get(idx).children[ci_pos];
                let c = self.arena.get(ci);
                let started = c.visits + c.virtual_loss;
                // Negamax: child.q() in child's POV. Negate to get parent's POV.
                let q_val = if started > 0 { -c.q() } else { fpu };
                let mut score = q_val + cpuct * c.prior * sqrt_n / (1 + started) as f32;
                // Bounds-aware биас (LC0 StickyEndgames):
                //   c.upper == -1 → child гарантированно проигрывает → parent выигрывает: бустим.
                //   c.lower == +1 → child гарантированно выигрывает → parent проигрывает: давим.
                if c.upper == -1 { score += 100.0; }
                else if c.lower == 1 { score -= 100.0; }
                // M-utility: предпочесть короткие выигрыши / длинные проигрыши.
                // Применяем только для посещённых детей (для unvisited m=0, нет данных).
                if m_active && started > 0 {
                    let dm = (M_SLOPE * (c.m - parent_m)).clamp(-M_CAP, M_CAP);
                    // q_val = -c.q() — оценка хода от лица parent'a.
                    //   q_val > 0 (ход выигрышный для parent) → sign(-q_val) = -1 → штрафуем dm>0 (длинный)
                    //   q_val < 0 (ход проигрышный) → sign(-q_val) = +1 → бонус dm>0 (затянуть)
                    let m_util = dm * (-q_val).signum() * q_val.abs();
                    score += m_util;
                }
                if !score.is_nan() && score > best { best = score; best_ci = ci; }
                if can_exit { break; }
                if started == 0 { can_exit = true; }
            }
            idx = best_ci;
        }
    }

    fn apply_vloss(&mut self, mut idx: usize, delta: i32) {
        loop {
            let n = self.arena.get_mut(idx);
            n.virtual_loss = if delta < 0 { (n.virtual_loss + delta).max(0) } else { n.virtual_loss + delta };
            match n.parent { Some(p) => idx = p, None => break }
        }
    }

    /// Возвращает total_count = сколько раз позиция leaf встречалась всего:
    ///   - в реальной истории игры (position_history включает корень)
    ///   - в path от ROOT (excl.) до leaf (incl., если leaf != root)
    ///
    /// Оптимизация: использует position_hash, сохранённые в узлах при expand.
    /// Старая версия проигрывала ходы вперёд от корня = O(P²) с board.apply_move.
    /// Новая = O(P) reads из arena.
    fn rep_count_at_leaf(&self, leaf_idx: usize, leaf_board: &Board) -> u32 {
        let leaf_hash = compute_board_hash(leaf_board);
        let history_count = self.position_history.iter()
            .filter(|&&h| h == leaf_hash).count() as u32;

        // Path count: считаем leaf только если он НЕ root (root уже в position_history,
        // не дубль-считаем). Затем walk UP: считаем intermediate-ноды (не root).
        //
        // Без проверки `leaf_idx == self.root` start=1 при первом expand'е root давал
        // total = history(1) + path(1) = 2 → root помечался как 2-fold терминал
        // → MCTS никогда не expand'ил детей → весь поиск ломался.
        let mut path_count = if leaf_idx == self.root { 0u32 } else { 1u32 };
        let mut cur = leaf_idx;
        while let Some(parent) = self.arena.get(cur).parent {
            if parent == self.root { break; }
            let pn = self.arena.get(parent);
            // position_hash установлен при expand. Если 0 → нераскрыт, не сравниваем.
            if pn.position_hash != 0 && pn.position_hash == leaf_hash {
                path_count += 1;
            }
            cur = parent;
        }
        history_count + path_count
    }

    fn is_repetition_at_leaf(&self, leaf_idx: usize, leaf_board: &Board) -> bool {
        self.rep_count_at_leaf(leaf_idx, leaf_board) >= 2
    }

    // board передаётся снаружи (уже восстановлен через board_at или хранится в pending_boards)
    fn expand(&mut self, idx: usize, board: &Board, policy: &[f32], add_noise: bool, rng: &mut u64) {
        // Сначала записываем position_hash — нужен для O(1) repetition check у потомков.
        let board_hash = compute_board_hash(board);
        self.arena.get_mut(idx).position_hash = board_hash;

        let legal = board.gen_legal();
        // Терминалы: натуральные (мат/пат/50-move/insufficient) — permanent,
        // 2-fold — path-dependent (LC0-style), нужна ревалидация после tree reuse.
        let natural_terminal = legal.is_empty()
            || board.halfmove_clock >= 100
            || board.is_insufficient_material();
        let twofold_terminal = !natural_terminal && self.is_repetition_at_leaf(idx, board);

        if natural_terminal || twofold_terminal {
            // Определяем результат от POV стороны на узле
            let side = board.side;
            let v: i8 = if legal.is_empty() {
                if board.in_check(side) { -1 } else { 0 }  // mate vs stalemate
            } else { 0 };  // правило ничьей
            let n = self.arena.get_mut(idx);
            n.is_terminal = true;
            n.is_expanded = true;
            n.terminal_kind = if natural_terminal { TERMINAL_KIND_NATURAL } else { TERMINAL_KIND_TWOFOLD };
            n.lower = v;
            n.upper = v;
            n.wl = v as f32;
            n.d = if v == 0 { 1.0 } else { 0.0 };
            return;
        }
        let n = legal.len();
        let side = board.side;
        let mut priors: Vec<f32> = legal.iter().map(|&(f, t, p)| {
            let pi = Board::move_to_idx(f, t, p, side);
            if pi < policy.len() { policy[pi] } else { 1e-8 }
        }).collect();
        let sum: f32 = priors.iter().sum();
        if sum <= 1e-12 { priors.iter_mut().for_each(|x| *x = 1.0/n as f32); }
        else { priors.iter_mut().for_each(|x| *x /= sum); }
        if add_noise {
            let dynamic_alpha = (10.0_f64 / n as f64).max(0.1);
            let noise = dirichlet_noise(dynamic_alpha, n, rng);
            for (p, &nd) in priors.iter_mut().zip(noise.iter()) {
                *p = (1.0 - DIRICHLET_EPS_V as f32) * *p + DIRICHLET_EPS_V as f32 * nd as f32;
            }
        }
        // Дочерние узлы: храним только ход и prior — Board не нужен
        let parent_side = board.side;
        let child_side = (1 - parent_side) as u8;
        let mut child_ids = Vec::with_capacity(n);
        for (i, &(f, t, p)) in legal.iter().enumerate() {
            let m = (f << 10) | (t << 3) | p.map(|pr| pr as u32 + 1).unwrap_or(0);
            let ci = self.arena.add(MctsNode::new(m, priors[i], child_side, Some(idx)));
            child_ids.push(ci);
        }
        // Sort children by prior descending (LC0 SortEdges).
        // Enables early-exit in select(): first unvisited child is always best
        // among all unvisited (highest P), so we can stop after seeing 2 unvisited.
        let mut pairs: Vec<(usize, f32)> = child_ids.iter()
            .map(|&ci| (ci, self.arena.get(ci).prior))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.arena.get_mut(idx).children = pairs.into_iter().map(|(ci, _)| ci).collect();
        self.arena.get_mut(idx).is_expanded = true;
    }

    fn backup(&mut self, mut idx: usize, value: f32, draw_prob: f32, mlh_ply: f32) {
        // Running averages (LC0 FinalizeScoreUpdate):
        //   wl += (v*sign - wl) / (n + 1)            — знак чередуется (negamax)
        //   d  += (d_leaf - d) / (n + 1)             — D не флипает (ничья симметрична)
        //   m  += (mlh_leaf + plies_from_leaf - m) / (n + 1)
        //     ↑ КРИТИЧЕСКИЙ ФИКС (Gemini): на КАЖДОМ уровне вверх от листа добавляем 1.
        //     Лист — m_ply от leaf. Родитель — m_ply+1 (на 1 ход раньше = на 1 ход больше осталось).
        //     Прародитель — m_ply+2. И т.д.
        let mut sign = 1.0f32;
        let mut plies_from_leaf = 0.0f32;
        loop {
            let n = self.arena.get_mut(idx);
            n.visits += 1;
            n.wl += (value * sign - n.wl) / n.visits as f32;
            n.d  += (draw_prob - n.d) / n.visits as f32;
            n.m  += (mlh_ply + plies_from_leaf - n.m) / n.visits as f32;
            n.virtual_loss = (n.virtual_loss - VIRTUAL_LOSS_V).max(0);
            sign *= -1.0;
            plies_from_leaf += 1.0;
            match n.parent { Some(p) => idx = p, None => break }
        }
    }

    /// Bounds propagation от только что доказанного терминала вверх по дереву (LC0 StickyEndgames).
    /// Когда узел стал терминальным/получил тайтер bounds — родитель может тоже доказаться.
    /// new_lower_parent = max over children of -child.upper  (лучший гарантированный результат)
    /// new_upper_parent = max over children of -child.lower  (лучший возможный результат)
    /// Если new_lower == new_upper → родитель доказан → ставим терминальным и продолжаем выше.
    fn propagate_bounds_from(&mut self, from_idx: usize) {
        let mut cur = from_idx;
        while let Some(parent) = self.arena.get(cur).parent {
            if !self.maybe_set_bounds(parent) { break; }
            cur = parent;
        }
    }

    fn maybe_set_bounds(&mut self, parent: usize) -> bool {
        if self.arena.get(parent).is_terminal { return false; }
        let n_ch = self.arena.get(parent).children.len();
        if n_ch == 0 { return false; }

        let mut new_lower: i8 = -1;
        let mut new_upper: i8 = -1;
        for ci_pos in 0..n_ch {
            let ci = self.arena.get(parent).children[ci_pos];
            let c = self.arena.get(ci);
            let pl = -c.upper;  // worst-case parent value via this child
            let pu = -c.lower;  // best-case parent value via this child
            if pl > new_lower { new_lower = pl; }
            if pu > new_upper { new_upper = pu; }
        }

        let (old_wl, old_d, old_lower, old_upper, p_visits) = {
            let p = self.arena.get(parent);
            (p.wl, p.d, p.lower, p.upper, p.visits)
        };
        if new_lower == old_lower && new_upper == old_upper { return false; }

        let became_terminal = new_lower == new_upper;
        let new_wl = new_lower as f32;
        let new_d = if new_lower == 0 { 1.0 } else { 0.0 };

        let p_mut = self.arena.get_mut(parent);
        p_mut.lower = new_lower;
        p_mut.upper = new_upper;
        if became_terminal {
            p_mut.is_terminal = true;
            p_mut.wl = new_wl;
            p_mut.d  = new_d;

            // LC0 AdjustForTerminal (node.cc:368):
            // Прежнее p.wl было running-average через NN-оценки. Теперь это значение
            // доказано как `new_wl` (точно). Все предки усреднили этот узел через
            // p_visits визитов с устаревшим old_wl. Применяем correction вверх.
            // Формула: ancestor.wl += sign * p_visits * (new_wl - old_wl) / ancestor.visits
            // Знак alterates (negamax). D не флипает знак.
            if p_visits > 0 {
                let v_delta = (new_wl - old_wl) * p_visits as f32;
                let d_delta = (new_d  - old_d ) * p_visits as f32;
                let mut sign = -1.0f32;  // первый предок в перевёрнутом POV
                let mut cur = parent;
                while let Some(gp) = self.arena.get(cur).parent {
                    let stop = self.arena.get(gp).is_terminal;
                    let gn = self.arena.get_mut(gp);
                    if gn.visits > 0 && !stop {
                        let n_inv = 1.0 / gn.visits as f32;
                        gn.wl = (gn.wl + sign * v_delta * n_inv).clamp(-1.0, 1.0);
                        gn.d  = (gn.d  + d_delta * n_inv).clamp(0.0, 1.0);
                    }
                    if stop { break; }
                    cur = gp;
                    sign *= -1.0;
                }
            }
        }
        true  // bounds сдвинулись — продолжаем пропагацию вверх
    }

    // Early stopping — если лучший ход математически не может быть догнан
    // вторым за оставшиеся симуляции, пропускаем игру.
    // ВАЖНО: срабатывает только если корень уже раскрыт И набрал достаточно визитов.
    // Без проверки root.visits дерево пропускалось ДО первой симуляции → нулевые policy.
    fn best_move_is_decided(&self, sims_remaining: i32) -> bool {
        let root = self.arena.get(self.root);
        // Не трогаем нераскрытые деревья и деревья с менее чем 2 детьми
        if !root.is_expanded || root.children.len() < 2 { return false; }
        // Нужен минимальный порог визитов чтобы статистика была значимой
        if root.visits < 10 { return false; }
        let mut best = 0i32;
        let mut second = 0i32;
        for &ci in &root.children {
            let v = self.arena.get(ci).visits;
            if v > best { second = best; best = v; }
            else if v > second { second = v; }
        }
        second + sims_remaining < best
    }

    // Версия с переиспользуемым буфером — без Vec<Vec<f32>> аллокаций
    fn collect_leaves_into_buf(&mut self, parallel: usize, _rng: &mut u64) -> usize {
        self.pending.clear();
        self.pending_boards.clear();
        self.leaf_tensor_buf.clear();
        let mut count = 0usize;
        for _ in 0..parallel {
            if let Some(leaf) = self.select() {
                if self.arena.get(leaf).is_terminal { continue; }
                // Защита от дубликатов: select() возвращает один и тот же нераскрытый узел
                // повторно (vloss не помогает для unexpanded — мы не доходим до PUCT-цикла).
                // Без этой проверки visits раздуваются + GPU считает одинаковые позиции.
                if self.pending.contains(&leaf) { continue; }
                self.apply_vloss(leaf, VIRTUAL_LOSS_V);
                let (leaf_board, history) = self.board_with_history_at(leaf);
                // Rep flag только для текущей (leaf) позиции — самая важная информация.
                // Старые slot'ы оставляем 0 для упрощения (LC0 V2 тоже их часто пропускает).
                let mut rep_flags = vec![false; history.len()];
                if !history.is_empty() && self.rep_count_at_leaf(leaf, &leaf_board) >= 2 {
                    rep_flags[0] = true;
                }
                let tensor = boards_to_tensor(
                    &history, &rep_flags,
                    leaf_board.side, leaf_board.halfmove_clock, leaf_board.castling,
                );
                self.leaf_tensor_buf.extend_from_slice(&tensor);
                self.pending_boards.push(leaf_board);
                self.pending.push(leaf);
                count += 1;
            }
        }
        count
    }

    // Старая версия — оставлена для совместимости
    fn collect_leaves(&mut self, parallel: usize, _rng: &mut u64) -> Vec<Vec<f32>> {
        self.pending.clear();
        self.pending_boards.clear();
        let mut tensors = Vec::new();
        for _ in 0..parallel {
            if let Some(leaf) = self.select() {
                if self.arena.get(leaf).is_terminal { continue; }
                self.apply_vloss(leaf, VIRTUAL_LOSS_V);
                let (leaf_board, history) = self.board_with_history_at(leaf);
                let mut rep_flags = vec![false; history.len()];
                if !history.is_empty() && self.rep_count_at_leaf(leaf, &leaf_board) >= 2 {
                    rep_flags[0] = true;
                }
                tensors.push(boards_to_tensor(
                    &history, &rep_flags,
                    leaf_board.side, leaf_board.halfmove_clock, leaf_board.castling,
                ));
                self.pending_boards.push(leaf_board);
                self.pending.push(leaf);
            }
        }
        tensors
    }

    fn apply_inference(&mut self, policies: &[Vec<f32>], values: &[f32], draws: &[f32], mlhs: &[f32], rng: &mut u64) {
        let pending = std::mem::take(&mut self.pending);
        let boards  = std::mem::take(&mut self.pending_boards);
        for (i, leaf) in pending.into_iter().enumerate() {
            if i >= policies.len() { break; }
            self.apply_vloss(leaf, -VIRTUAL_LOSS_V);
            let is_root = leaf == self.root;
            let was_unexpanded = !self.arena.get(leaf).is_expanded;
            if was_unexpanded {
                if let Some(board) = boards.get(i) {
                    self.expand(leaf, board, &policies[i], is_root, rng);
                }
            }
            // Терминал: expand уже выставил wl/d/bounds. m_terminal = 0 (игра уже окончена).
            // Не-терминал: используем NN value/draw/mlh. mlh из сети ∈ [0,1] → умножаем на NORM=200 → PLY.
            let leaf_terminal = self.arena.get(leaf).is_terminal;
            let (v, d, m_ply) = if leaf_terminal {
                let n = self.arena.get(leaf);
                (n.wl, n.d, 0.0_f32)  // терминал = 0 ply осталось
            } else {
                let dv = if i < draws.len() { draws[i].clamp(0.0, 1.0) } else { 0.0 };
                let mv = if i < mlhs.len() { mlhs[i].clamp(0.0, 1.0) * MLH_PLY_NORM } else { 0.0 };
                (values[i], dv, mv)
            };
            self.backup(leaf, v, d, m_ply);
            if was_unexpanded && leaf_terminal {
                self.propagate_bounds_from(leaf);
            }
        }
    }

    // Версия без Vec<Vec<f32>> — принимает плоский срез политик
    fn apply_inference_flat(&mut self, pol_flat: &[f32], policy_size: usize,
                            values: &[f32], draws: &[f32], mlhs: &[f32], rng: &mut u64) {
        let pending = std::mem::take(&mut self.pending);
        let boards  = std::mem::take(&mut self.pending_boards);
        for (i, leaf) in pending.into_iter().enumerate() {
            if i >= values.len() { break; }
            self.apply_vloss(leaf, -VIRTUAL_LOSS_V);
            let is_root = leaf == self.root;
            let was_unexpanded = !self.arena.get(leaf).is_expanded;
            if was_unexpanded {
                let start = i * policy_size;
                let end = start + policy_size;
                if end <= pol_flat.len() {
                    if let Some(board) = boards.get(i) {
                        self.expand(leaf, board, &pol_flat[start..end], is_root, rng);
                    }
                }
            }
            let leaf_terminal = self.arena.get(leaf).is_terminal;
            let (v, d, m_ply) = if leaf_terminal {
                let n = self.arena.get(leaf);
                (n.wl, n.d, 0.0_f32)
            } else {
                let dv = if i < draws.len() { draws[i].clamp(0.0, 1.0) } else { 0.0 };
                let mv = if i < mlhs.len() { mlhs[i].clamp(0.0, 1.0) * MLH_PLY_NORM } else { 0.0 };
                (values[i], dv, mv)
            };
            self.backup(leaf, v, d, m_ply);
            if was_unexpanded && leaf_terminal {
                self.propagate_bounds_from(leaf);
            }
        }
                            }

                            fn get_policy(&self) -> Vec<f32> {
                                let root = self.arena.get(self.root);
                                let total: i32 = root.children.iter().map(|&ci| self.arena.get(ci).visits).sum();
                                let mut pol = vec![0.0f32; POLICY_SIZE_MCTS];
                                if total > 0 {
                                    let side = self.root_board.side;
                                    for &ci in &root.children {
                                        let c = self.arena.get(ci);
                                        let m = c.move_from_parent;
                                        let f = (m >> 10) & 0x7F;
                                        let t = (m >> 3) & 0x7F;
                                        let pv = m & 0b111;
                                        let p = if pv == 0 { None } else { Some((pv - 1) as usize) };
                                        // Канонический индекс (см. Board::move_to_idx).
                                        let idx = Board::move_to_idx(f, t, p, side);
                                        if idx < POLICY_SIZE_MCTS { pol[idx] = c.visits as f32 / total as f32; }
                                    }
                                }
                                pol
                            }

                            fn is_over(&mut self) -> bool {
                                if self.root_board.halfmove_clock >= 100 { return true; }
                                if self.root_board.is_insufficient_material() { return true; }
                                // Троекратное повторение: текущая позиция уже в history (push в make_move),
                                // 3-fold = эта позиция встречалась >= 3 раз.
                                let cur_hash = Self::board_hash(&self.root_board);
                                let repeats = self.position_history.iter().filter(|&&h| h == cur_hash).count();
                                if repeats >= 3 { return true; }
                                self.root_board.gen_legal().is_empty()
                            }

                            fn root_value(&self) -> f32 {
                                self.arena.get(self.root).wl
                            }

                            fn root_draw(&self) -> f32 {
                                self.arena.get(self.root).d
                            }

                            // Рекурсивно копирует поддерево из self.arena в new_arena (без Board).
                            fn copy_subtree(&self, old_idx: usize, new_arena: &mut Arena, new_parent: Option<usize>) -> usize {
                                let old_node = self.arena.get(old_idx);
                                let mut new_node = MctsNode::new(
                                    old_node.move_from_parent,
                                    old_node.prior,
                                    old_node.side,
                                    new_parent,
                                );
                                // Сохраняем всю накопленную статистику и кеши.
                                new_node.visits        = old_node.visits;
                                new_node.wl            = old_node.wl;
                                new_node.d             = old_node.d;
                                new_node.m             = old_node.m;
                                new_node.is_expanded   = old_node.is_expanded;
                                new_node.is_terminal   = old_node.is_terminal;
                                new_node.terminal_kind = old_node.terminal_kind;
                                new_node.lower         = old_node.lower;
                                new_node.upper         = old_node.upper;
                                new_node.position_hash = old_node.position_hash;
                                let new_idx = new_arena.add(new_node);
                                let children: Vec<usize> = old_node.children.clone();
                                let mut new_children = Vec::with_capacity(children.len());
                                for child_old in children {
                                    new_children.push(self.copy_subtree(child_old, new_arena, Some(new_idx)));
                                }
                                new_arena.get_mut(new_idx).children = new_children;
                                new_idx
                            }

                            fn make_move(&mut self, m_int: u32) {
                                // Применяем ход к root_board
                                let pv = m_int & 0b111;
                                let t  = (m_int >> 3) & 0x7F;
                                let f  = (m_int >> 10) & 0x7F;
                                let p  = if pv == 0 { None } else { Some((pv - 1) as usize) };
                                self.root_board.apply_move(f, t, p);
                                let new_side = self.root_board.side as u8;

                                let child_idx = self.arena.get(self.root).children.iter()
                                .copied()
                                .find(|&ci| self.arena.get(ci).move_from_parent == m_int);

                                // Tree GC — копируем только нужное поддерево в новую арену.
                                let mut new_arena = Arena::new(8192);
                                self.root = if let Some(ci) = child_idx {
                                    self.copy_subtree(ci, &mut new_arena, None)
                                } else {
                                    // Ход не был в дереве — создаём корень с нуля
                                    new_arena.add(MctsNode::new(m_int, 1.0, new_side, None))
                                };
                                self.arena = new_arena;
                                // Accumulating tree reuse: фиксируем унаследованные визиты —
                                // новый бюджет симуляций будет считаться поверх них.
                                self.move_start_visits = self.arena.get(self.root).visits;
                                self.pending.clear();
                                self.pending_boards.clear();
                                // Обновляем историю: при необратимом ходе (взятие/пешка) старые позиции не повторятся
                                let new_hash = Self::board_hash(&self.root_board);
                                if self.root_board.halfmove_clock == 0 {
                                    self.position_history.clear();
                                }
                                self.position_history.push(new_hash);
                                // board history (LC0 history planes): push новой позиции в front.
                                self.root_history.insert(0, self.root_board.clone());
                                if self.root_history.len() > HISTORY_LEN {
                                    self.root_history.truncate(HISTORY_LEN);
                                }
                                // LC0 EnsureNodeTwoFoldCorrectForDepth (search.cc:1532):
                                // После смещения корня ревалидируем path-dependent терминалы.
                                self.revalidate_after_root_shift();
                            }

                            /// Ревалидация терминалов после tree reuse (LC0 EnsureNodeTwoFoldCorrect):
                            ///   - Natural терминалы (мат/пат/50-move/insufficient) — permanent, оставляем.
                            ///   - 2-fold терминалы — переcчитываем rep_count для нового пути от ROOT.
                            ///     Если больше не повторение → откатываем флаг, re-expand.
                            ///   - Bounds-prop терминалы (terminal_kind=NONE+is_terminal=true) —
                            ///     консервативно сбрасываем bounds (будут переоткрыты следующими сим.).
                            fn revalidate_after_root_shift(&mut self) {
                                let n_nodes = self.arena.nodes.len();
                                for idx in 0..n_nodes {
                                    let (term, kind) = {
                                        let n = self.arena.get(idx);
                                        (n.is_terminal, n.terminal_kind)
                                    };
                                    if !term { continue; }
                                    if kind == TERMINAL_KIND_NATURAL { continue; }
                                    if kind == TERMINAL_KIND_TWOFOLD {
                                        // Реконструируем board + проверяем повторение
                                        let board = self.board_at(idx);
                                        if self.rep_count_at_leaf(idx, &board) >= 2 {
                                            continue;  // всё ещё повторение
                                        }
                                        // Не повторение больше — сбрасываем как leaf для re-expand.
                                        let n = self.arena.get_mut(idx);
                                        n.is_terminal = false;
                                        n.is_expanded = false;
                                        n.terminal_kind = TERMINAL_KIND_NONE;
                                        n.lower = -1;
                                        n.upper = 1;
                                        n.children.clear();
                                        // wl/d/visits сохраняем — будут уточнены новыми сим.
                                    } else {
                                        // kind == NONE + is_terminal=true → bounds-prop terminal.
                                        // Дети сохранены, флаг сбрасываем, на след.сим. может переоткрыться.
                                        let n = self.arena.get_mut(idx);
                                        n.is_terminal = false;
                                        n.lower = -1;
                                        n.upper = 1;
                                    }
                                }
                            }

                            /// Перенаносит Dirichlet noise на priors детей текущего корня.
                            /// Вызывается после make_move при tree reuse — иначе исследовательский
                            /// шум применяется только на ПЕРВОМ ходу всей партии.
                            fn renoise_root(&mut self, rng: &mut u64) {
                                let root_idx = self.root;
                                let children: Vec<usize> = self.arena.get(root_idx).children.clone();
                                let n = children.len();
                                if n == 0 { return; }
                                let dynamic_alpha = (10.0_f64 / n as f64).max(0.1);
                                let noise = dirichlet_noise(dynamic_alpha, n, rng);
                                for (i, ci) in children.iter().enumerate() {
                                    let p = self.arena.get(*ci).prior;
                                    let mixed = (1.0 - DIRICHLET_EPS_V as f32) * p
                                              + DIRICHLET_EPS_V as f32 * noise[i] as f32;
                                    self.arena.get_mut(*ci).prior = mixed;
                                }
                                // Re-sort children by NEW prior (LC0 SortEdges semantics).
                                // Без этого early-exit в select() работает с устаревшим порядком.
                                let mut pairs: Vec<(usize, f32)> = children.iter()
                                    .map(|&ci| (ci, self.arena.get(ci).prior))
                                    .collect();
                                pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                self.arena.get_mut(root_idx).children = pairs.into_iter().map(|(ci, _)| ci).collect();
                            }
}

/// BatchState хранит leaf_counts вместе с данными батча.
/// Это позволяет двойной буферизации применять политики к правильному батчу —
/// self.leaf_counts перезаписывается следующим collect_leaves, но BatchState нет.
struct BatchState {
    leaf_counts: Vec<usize>,
}

/// RustMCTS — батчевый MCTS для N игр одновременно.
#[pyclass]
pub struct RustMCTS {
    games: Vec<SingleMcts>,
    parallel_sims: usize,
    rng: u64,
    leaf_game_map: Vec<usize>,
    leaf_counts: Vec<usize>,
    prev_batch: Option<BatchState>,  // для корректной двойной буферизации
}

#[pymethods]
impl RustMCTS {
    #[new]
    pub fn new(engines: Vec<PyRef<CapablancaEngine>>, parallel_sims: usize) -> Self {
        let games = engines.iter().map(|e|
            SingleMcts::new_with_history(e.board.clone(), e.board_history.clone())
        ).collect();
        // Сид из системного времени — чтобы параллельные RustMCTS (fsf/lagged
        // создают по объекту на игру) не получали одинаковый Dirichlet noise.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdeadbeefcafe1234u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(0xbf58476d1ce4e5b9);
        RustMCTS { games, parallel_sims, rng: seed,
            leaf_game_map: Vec::new(), leaf_counts: Vec::new(), prev_batch: None }
    }

    /// Собирает листья для inference.
    /// Возвращает 2D NumPy массив формы (N, TOTAL_INPUT_PLANES*80) = (N, 11120) с history planes.
    /// target_sims_per_game = сколько всего симуляций планируется для каждой игры
    /// (для честной оценки sims_remaining в best_move_is_decided).
    #[pyo3(signature = (target_sims_per_game=0))]
    pub fn collect_leaves<'py>(&mut self, py: Python<'py>, target_sims_per_game: i32) -> Bound<'py, PyArray2<f32>> {
        self.leaf_game_map.clear();
        let mut new_counts = vec![0usize; self.games.len()];
        let mut flat: Vec<f32> = Vec::new();
        let mut total = 0usize;

        for (g, game) in self.games.iter_mut().enumerate() {
            if game.is_over() { continue; }
            // Accumulating tree reuse: остаток считаем от sims СДЕЛАННЫХ ЗА ТЕКУЩИЙ ХОД,
            // а не от полного root.visits. Иначе унаследованные через tree reuse визиты
            // искусственно занижают sims_remaining → best_move_is_decided обрывает поиск
            // преждевременно ("душит"). target_sims_per_game=0 = не использовать early-stop.
            let visits_so_far = game.arena.get(game.root).visits;
            if target_sims_per_game > 0 {
                let sims_done_this_move = (visits_so_far - game.move_start_visits).max(0);
                let sims_remaining = (target_sims_per_game - sims_done_this_move).max(0);
                if game.best_move_is_decided(sims_remaining) { continue; }
            }
            // Используем буферизованный сбор без лишних аллокаций Vec<Vec<f32>>
            let count = game.collect_leaves_into_buf(self.parallel_sims, &mut self.rng);
            new_counts[g] = count;
            for _ in 0..count { self.leaf_game_map.push(g); }
            total += count;
            flat.extend_from_slice(&game.leaf_tensor_buf);
        }

        // Сохраняем leaf_counts ЭТОГО батча в prev_batch ПЕРЕД перезаписью self.leaf_counts.
        // apply_inference_buffered берёт counts из Python (переданные после collect_leaves),
        // поэтому политики всегда идут к правильным играм при двойной буферизации.
        self.prev_batch = Some(BatchState { leaf_counts: new_counts.clone() });
        self.leaf_counts = new_counts;

        // Размер одного тензора = TOTAL_INPUT_PLANES * 80 (139 * 80 = 11120 после history planes).
        let cols = TOTAL_INPUT_PLANES * 80;
        if total == 0 {
            Array2::<f32>::zeros((0, cols)).into_pyarray(py).into()
        } else {
            Array2::from_shape_vec((total, cols), flat)
            .expect("collect_leaves: shape mismatch")
            .into_pyarray(py)
            .into()
        }
    }

    /// Хеши текущих pending листьев (по 8 байт на лист, в порядке как collect_leaves).
    /// Используется для NN transposition cache: ключ позиции = 64-битный хеш,
    /// не зависит от истории (LC0 kCacheHistoryLength=0). Раньше Python кешировал по
    /// байтам всего тензора (44KB ключ) — медленно + одна позиция с разной историей
    /// давала разные ключи → почти 0% hit rate.
    pub fn get_leaf_hashes(&self) -> Vec<u64> {
        let mut hashes = Vec::new();
        for game in &self.games {
            for board in &game.pending_boards {
                hashes.push(compute_board_hash(board));
            }
        }
        hashes
    }

    /// Применяет результаты GPU inference к деревьям.
    /// Принимает NumPy массивы напрямую — нулевой overhead на сериализацию.
    /// policies: shape (N, 7000) f32
    /// values:   shape (N,)      f32 — Q = P(W) - P(L)
    /// draws:    shape (N,)      f32 — P(D), для D-head tracking в узлах
    pub fn apply_inference(
        &mut self,
        policies: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
        draws: PyReadonlyArray1<f32>,
        mlhs: PyReadonlyArray1<f32>,
    ) {
        let pol_flat = policies.as_slice().expect("policies must be C-contiguous");
        let val     = values.as_slice().expect("values must be contiguous");
        let drw     = draws.as_slice().expect("draws must be contiguous");
        let mlh     = mlhs.as_slice().expect("mlhs must be contiguous");

        let shape       = policies.shape();
        let n_leaves    = shape[0];
        let policy_size = shape[1];

        let total: usize = self.leaf_counts.iter().sum();
        if total != n_leaves || val.len() != n_leaves || drw.len() != n_leaves || mlh.len() != n_leaves {
            eprintln!(
                "apply_inference: size mismatch (leaf_counts.sum={} n_leaves={} val.len={} drw.len={} mlh.len={}) — resetting vloss",
                total, n_leaves, val.len(), drw.len(), mlh.len()
            );
            let n = self.games.len();
            self.drain_pending_vloss(0, n);
            return;
        }

        let mut offset = 0;
        let mut break_at: Option<usize> = None;
        let counts = std::mem::take(&mut self.leaf_counts);
        for (g, &count) in counts.iter().enumerate() {
            if count == 0 { continue; }
            let start = offset * policy_size;
            let end   = (offset + count) * policy_size;
            if end > pol_flat.len() {
                eprintln!("apply_inference: pol_flat overflow at game {} — resetting vloss for remaining", g);
                break_at = Some(g);
                break;
            }
            let rng = &mut self.rng;
            self.games[g].apply_inference_flat(
                &pol_flat[start..end], policy_size,
                &val[offset..offset + count],
                &drw[offset..offset + count],
                &mlh[offset..offset + count],
                rng,
            );
            offset += count;
        }
        self.leaf_counts = counts;
        if let Some(g) = break_at {
            let n = self.games.len();
            self.drain_pending_vloss(g, n);
        }
    }

    /// Возвращает leaf_counts текущего батча — Python сохраняет и передаёт
    /// в apply_inference_buffered при двойной буферизации.
    pub fn get_current_batch_counts(&self) -> Vec<usize> {
        self.leaf_counts.clone()
    }

    /// Снимает накопленный virtual_loss и очищает pending для игр в диапазоне [from, to).
    /// Нужен на error-paths: иначе vloss остался бы навсегда и поиск был бы отравлен.
    fn drain_pending_vloss(&mut self, from: usize, to: usize) {
        let end = to.min(self.games.len());
        for g in from..end {
            let game = &mut self.games[g];
            let pending = std::mem::take(&mut game.pending);
            game.pending_boards.clear();
            for leaf in pending {
                game.apply_vloss(leaf, -VIRTUAL_LOSS_V);
            }
        }
    }

    /// apply_inference с явным batch_counts для правильной двойной буферизации.
    pub fn apply_inference_buffered(
        &mut self,
        policies: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
        draws: PyReadonlyArray1<f32>,
        mlhs: PyReadonlyArray1<f32>,
        batch_counts: Vec<usize>,
    ) {
        let pol_flat    = policies.as_slice().expect("policies must be C-contiguous");
        let val         = values.as_slice().expect("values must be contiguous");
        let drw         = draws.as_slice().expect("draws must be contiguous");
        let mlh         = mlhs.as_slice().expect("mlhs must be contiguous");
        let shape       = policies.shape();
        let n_leaves    = shape[0];
        let policy_size = shape[1];
        let total: usize = batch_counts.iter().sum();

        if total != n_leaves || val.len() != n_leaves || drw.len() != n_leaves || mlh.len() != n_leaves {
            // Рассинхрон Python/Rust state machine. Тихий return оставлял бы vloss
            // на pending листьях навсегда → дерево поиска отравлено до конца партии.
            eprintln!(
                "apply_inference_buffered: size mismatch (batch_counts.sum={} n_leaves={} val.len={} drw.len={} mlh.len={}) — resetting vloss",
                total, n_leaves, val.len(), drw.len(), mlh.len()
            );
            let n = self.games.len();
            self.drain_pending_vloss(0, n);
            return;
        }

        let mut offset = 0;
        let mut break_at: Option<usize> = None;
        for (g, &count) in batch_counts.iter().enumerate() {
            if count == 0 { continue; }
            let start = offset * policy_size;
            let end   = (offset + count) * policy_size;
            if end > pol_flat.len() {
                eprintln!(
                    "apply_inference_buffered: pol_flat overflow at game {} (end={} > len={}) — resetting vloss for remaining",
                    g, end, pol_flat.len()
                );
                break_at = Some(g);
                break;
            }
            let rng = &mut self.rng;
            self.games[g].apply_inference_flat(
                &pol_flat[start..end], policy_size,
                &val[offset..offset + count],
                &drw[offset..offset + count],
                &mlh[offset..offset + count],
                rng,
            );
            offset += count;
        }
        if let Some(g) = break_at {
            let n = self.games.len();
            self.drain_pending_vloss(g, n);
        }
    }

    /// Финальные policy-векторы из visit counts.
    pub fn get_policies(&self) -> Vec<Vec<f32>> {
        self.games.iter().map(|g| g.get_policy()).collect()
    }

    /// Value оценки корней (Q = P(W) - P(L)).
    pub fn get_values(&self) -> Vec<f32> {
        self.games.iter().map(|g| g.root_value()).collect()
    }

    /// Draw-вероятности корней (D = P(Draw)). Используется для WDL-based resign:
    /// P(L) = (1 - Q - D) / 2 — точнее, чем порог по Q (различает "верная ничья" и "проигрыш").
    pub fn get_draws(&self) -> Vec<f32> {
        self.games.iter().map(|g| g.root_draw()).collect()
    }

    /// Статус завершения игр.
    pub fn games_over(&mut self) -> Vec<bool> {
        self.games.iter_mut().map(|g| g.is_over()).collect()
    }

    /// Применяет ход к конкретной игре (для tree reuse).
    /// После переноса корня — добавляем свежий Dirichlet noise, иначе exploration
    /// падает до нуля начиная со 2-го полухода партии.
    pub fn make_move(&mut self, game_idx: usize, m_int: u32) {
        if game_idx < self.games.len() {
            self.games[game_idx].make_move(m_int);
            let rng = &mut self.rng;
            self.games[game_idx].renoise_root(rng);
            // root изменился → старый snapshot невалиден
            self.games[game_idx].kld_reset();
        }
    }

    /// KLD-early-exit gate. Считает max KL(prev || curr) среди всех игр
    /// и СРАЗУ сохраняет current как новый snapshot.
    ///
    /// Возвращает f32::INFINITY если у каких-то игр нет snapshot — первый вызов
    /// после make_move/init всегда возвращает +inf (Python должен пропустить gate
    /// на первой проверке и использовать только результаты второго+ вызова).
    ///
    /// Вся compute проходит в Rust → marshalling cost = 1 float вместо 7000*N_games.
    pub fn kld_snapshot_and_check(&mut self) -> f32 {
        let mut max_kl = 0.0_f32;
        let mut has_prev = true;
        for g in &mut self.games {
            let kl = g.kld_compute_gain();
            if kl.is_infinite() {
                has_prev = false;
            } else if kl > max_kl {
                max_kl = kl;
            }
            g.kld_take_snapshot();
        }
        if has_prev { max_kl } else { f32::INFINITY }
    }

    /// Сбросить snapshots всем играм. Нужно вручную если внешний код менял корень.
    pub fn kld_reset_all(&mut self) {
        for g in &mut self.games { g.kld_reset(); }
    }

    pub fn num_games(&self) -> usize { self.games.len() }
    pub fn last_batch_size(&self) -> usize { self.leaf_game_map.len() }
}
