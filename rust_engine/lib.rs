// src/lib.rs — Capablanca Chess Engine (10x8 board)
// Pieces: P=0, N=1, B=2, R=3, Q=4, A=5(Archbishop=B+N), C=6(Chancellor=R+N), K=7
//
// Board layout (bit index):
//   rank 0 (bottom/white) = bits 0..9
//   rank 7 (top/black)    = bits 70..79
//   square(file, rank) = rank * 10 + file
//
// Starting position (Capablanca Standard):
//   White: R N A B Q K B C N R  (rank 0, files 0-9)
//   Black: r n a b q k b c n r  (rank 7, files 0-9)
//   Pawns on ranks 1 (white) and 6 (black)

use pyo3::prelude::*;

type BB = u128; // 128-bit bitboard, only bits 0..79 used

const BOARD_MASK: BB = (1u128 << 80) - 1;

fn file_mask(f: u32) -> BB {
    let mut m: BB = 0;
    for r in 0..8u32 {
        m |= 1u128 << (r * 10 + f);
    }
    m
}

fn rank_mask(r: u32) -> BB {
    ((1u128 << 10) - 1) << (r * 10)
}

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
        let prev_file = ((current - delta) % 10 + 10) as u32 % 10;
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

fn bishop_attacks(sq: u32, occ: BB) -> BB {
    ray_attacks(sq, occ, 11) | ray_attacks(sq, occ, 9) | ray_attacks(sq, occ, -9) | ray_attacks(sq, occ, -11)
}

fn rook_attacks(sq: u32, occ: BB) -> BB {
    ray_attacks(sq, occ, 10) | ray_attacks(sq, occ, -10) | ray_attacks(sq, occ, 1) | ray_attacks(sq, occ, -1)
}

fn queen_attacks(sq: u32, occ: BB) -> BB { bishop_attacks(sq, occ) | rook_attacks(sq, occ) }
fn archbishop_attacks(sq: u32, occ: BB) -> BB { bishop_attacks(sq, occ) | knight_attacks(sq) }
fn chancellor_attacks(sq: u32, occ: BB) -> BB { rook_attacks(sq, occ) | knight_attacks(sq) }

fn king_attacks(sq: u32) -> BB {
    let b: BB = 1u128 << sq;
    let not_a = not_file_a();
    let not_j = not_file_j();
    ((b << 10) | (b >> 10) | ((b << 1) & not_a) | ((b >> 1) & not_j) | ((b << 11) & not_a) | ((b << 9) & not_j) | ((b >> 9) & not_a) | ((b >> 11) & not_j)) & BOARD_MASK
}

fn white_pawn_attacks(pawns: BB) -> BB {
    let not_a = not_file_a();
    let not_j = not_file_j();
    ((pawns << 11) & not_a) | ((pawns << 9) & not_j)
}

fn black_pawn_attacks(pawns: BB) -> BB {
    let not_a = not_file_a();
    let not_j = not_file_j();
    ((pawns >> 9) & not_a) | ((pawns >> 11) & not_j)
}

const PAWN: usize = 0;
const KNIGHT: usize = 1;
const BISHOP: usize = 2;
const ROOK: usize = 3;
const QUEEN: usize = 4;
const ARCH: usize = 5;
const CHANC: usize = 6;
const KING: usize = 7;

#[derive(Clone)]
pub struct Board {
    pieces: [[BB; 8]; 2],
    side: usize,
    castling: u8,
    ep_square: Option<u8>,
    halfmove_clock: u32,
    fullmove: u32,
}

impl Board {
    fn all_pieces(&self, color: usize) -> BB { self.pieces[color].iter().fold(0, |a, &b| a | b) }
    fn occupancy(&self) -> BB { self.all_pieces(0) | self.all_pieces(1) }

    // ИСПРАВЛЕННАЯ РАССТАНОВКА
    fn start() -> Self {
        let mut b = Board {
            pieces: [[0; 8]; 2],
            side: 0,
            castling: 0b1111,
            ep_square: None,
            halfmove_clock: 0,
            fullmove: 1,
        };

        // White pieces: R N A B Q K B C N R
        let white_back: [(usize, u32); 10] = [
            (ROOK,   0), (KNIGHT, 1), (ARCH,   2), (BISHOP, 3), (QUEEN,  4),
            (KING,   5), (BISHOP, 6), (CHANC,  7), (KNIGHT, 8), (ROOK,   9),
        ];
        for (pt, f) in white_back { b.pieces[0][pt] |= 1u128 << f; }
        b.pieces[0][PAWN] = rank_mask(1);

        // Black pieces: r n a b q k b c n r
        let black_back: [(usize, u32); 10] = [
            (ROOK,   0), (KNIGHT, 1), (ARCH,   2), (BISHOP, 3), (QUEEN,  4),
            (KING,   5), (BISHOP, 6), (CHANC,  7), (KNIGHT, 8), (ROOK,   9),
        ];
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
        let us = self.side;
        let them = 1 - us;
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
            for to in bb_iter(cap_r | cap_l) {
                let from = if cap_r & (1u128 << to) != 0 { to - 11 } else { to - 9 };
                add_pawn_move(from, to, us, &mut moves);
            }
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
            for to in bb_iter(cap_r | cap_l) {
                let from = if cap_r & (1u128 << to) != 0 { to + 9 } else { to + 11 };
                add_pawn_move(from, to, us, &mut moves);
            }
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
        let us = self.side;
        let occ = self.occupancy();
        let opp_att = self.attacks_by(1 - us);
        let back_rank = if us == 0 { 0u32 } else { 7u32 };
        let king_sq = back_rank * 10 + 5;
        if self.pieces[us][KING] & (1u128 << king_sq) == 0 || (opp_att & (1u128 << king_sq) != 0) { return; }
        if self.castling & (1 << (us * 2)) != 0 {
            let sq6 = 1u128 << (back_rank * 10 + 6);
            let sq7 = 1u128 << (back_rank * 10 + 7);
            if occ & (sq6 | sq7) == 0 && (self.pieces[us][ROOK] & (1u128 << (back_rank * 10 + 9)) != 0) && (opp_att & (sq6 | sq7) == 0) { moves.push((king_sq, back_rank * 10 + 7, None)); }
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
        let mut moving_piece = PAWN;
        for p in 0..8 { if self.pieces[us][p] & from_bb != 0 { moving_piece = p; break; } }
        for p in 0..8 { self.pieces[them][p] &= !to_bb; }
        if moving_piece == PAWN {
            if let Some(ep) = self.ep_square { if to == ep as u32 { self.pieces[them][PAWN] &= !(1u128 << (if us == 0 { to - 10 } else { to + 10 })); } }
        }
        self.pieces[us][moving_piece] &= !from_bb;
        self.pieces[us][moving_piece] |= to_bb;
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
        let rook_sqs = [(0, 0), (9, 1), (70, 2), (79, 3)];
        for (sq, bit) in rook_sqs { if from == sq as u32 || to == sq as u32 { self.castling &= !(1 << bit); } }
        self.ep_square = None;
        if moving_piece == PAWN {
            if us == 0 && from + 20 == to { self.ep_square = Some((from + 10) as u8); }
            else if us == 1 && from == to + 20 { self.ep_square = Some((to + 10) as u8); }
        }
        self.halfmove_clock = if moving_piece == PAWN { 0 } else { self.halfmove_clock + 1 };
        if us == 1 { self.fullmove += 1; }
        self.side = them;
    }

    fn to_tensor(&self) -> Vec<f32> {
        let mut t = vec![0.0f32; 20 * 80];
        for c in 0..2 { for p in 0..8 { for sq in bb_iter(self.pieces[c][p]) { t[(c * 8 + p) * 80 + sq as usize] = 1.0; } } }
        let side_val = if self.side == 0 { 1.0 } else { 0.0 };
        for i in 0..80 { t[16 * 80 + i] = side_val; }
        for i in 0..4 { t[17 * 80 + i] = ((self.castling >> i) & 1) as f32; }
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
                6400 + pi * 80 + to as usize
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

const FILES: [char; 10] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
fn sq_to_uci(sq: u32) -> String { format!("{}{}", FILES[(sq % 10) as usize], sq / 10 + 1) }
fn uci_to_sq(s: &str) -> Option<u32> {
    if s.len() < 2 { return None; }
    let first_char = s.chars().next()?;
    let f = FILES.iter().position(|&c| c == first_char)? as u32;
    let r = s.chars().nth(1)?.to_digit(10)? - 1;
    Some(r * 10 + f)
}
fn promo_char(p: usize) -> char { match p { ARCH => 'a', CHANC => 'c', ROOK => 'r', BISHOP => 'b', KNIGHT => 'n', _ => 'q' } }
fn char_to_promo(c: char) -> usize { match c { 'a' => ARCH, 'c' => CHANC, 'r' => ROOK, 'b' => BISHOP, 'n' => KNIGHT, _ => QUEEN } }
fn move_to_uci(f: u32, t: u32, p: Option<usize>) -> String {
    let mut s = format!("{}{}", sq_to_uci(f), sq_to_uci(t));
    if let Some(pr) = p { s.push(promo_char(pr)); }
    s
}
fn uci_to_move(s: &str) -> Option<(u32, u32, Option<usize>)> {
    if s.len() < 4 { return None; }
    let f = uci_to_sq(&s[0..2])?; let t = uci_to_sq(&s[2..4])?;
    let p = if s.len() >= 5 { Some(char_to_promo(s.chars().nth(4)?)) } else { None };
    Some((f, t, p))
}

#[pyclass] #[derive(Clone)] pub struct CapablancaEngine { board: Board }
#[pymethods]
impl CapablancaEngine {
    #[new]
    pub fn new() -> Self {
        CapablancaEngine { board: Board::start() }
    }

    // Тот самый метод copy, который мы пропустили
    pub fn copy(&self) -> Self {
        self.clone()
    }

    pub fn get_board_tensor(&self) -> Vec<f32> {
        self.board.to_tensor()
    }

    pub fn get_legal_moves(&self) -> Vec<String> {
        self.board.gen_legal().iter().map(|&(f, t, p)| move_to_uci(f, t, p)).collect()
    }

    pub fn make_move(&mut self, m: &str) -> bool {
        if let Some((f, t, p)) = uci_to_move(m) {
            if self.board.gen_legal().contains(&(f, t, p)) {
                self.board.apply_move(f, t, p);
                return true;
            }
        }
        false
    }

    pub fn is_game_over(&self) -> bool {
        self.board.gen_legal().is_empty() || self.board.halfmove_clock >= 100
    }

    // Метод для определения результата игры (нужен для MCTS)
    pub fn game_result(&self) -> f32 {
        if self.board.halfmove_clock >= 100 { return 0.0; }
        if self.board.gen_legal().is_empty() {
            if self.board.in_check(self.board.side) {
                // Если текущий игрок в шаху и нет ходов — это мат
                return if self.board.side == 0 { -1.0 } else { 1.0 };
            }
            return 0.0; // Пат
        }
        0.0 // Игра продолжается
    }

    pub fn is_stalemate(&self) -> bool {
        self.board.gen_legal().is_empty() && !self.board.in_check(self.board.side)
    }

    pub fn side_to_move(&self) -> usize {
        self.board.side
    }

    #[staticmethod]
    pub fn policy_size() -> usize { 6880 }

    pub fn move_to_policy_idx(&self, m: &str) -> Option<usize> {
        uci_to_move(m).map(|(f, t, p)| Board::move_to_idx(f, t, p))
    }
}

#[pymodule] fn capablanca_engine(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<CapablancaEngine>()?; Ok(()) }
