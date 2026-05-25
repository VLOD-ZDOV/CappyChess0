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

// Pre-computed attack tables — computed once at startup.
// Speeds up gen_pseudo_legal by 2-3x: no mask recomputation on each call.
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

fn white_pawn_attacks(pawns: BB) -> BB { ((pawns & not_file_j()) << 11) | ((pawns & not_file_a()) << 9) } // FIX: pre-shift masks
fn black_pawn_attacks(pawns: BB) -> BB { ((pawns & not_file_j()) >> 9) | ((pawns & not_file_a()) >> 11) } // FIX: pre-shift masks

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
            let cap_r = ((pawns & not_file_j()) << 11) & their_pieces; // FIX: pre-shift mask
            let cap_l = ((pawns & not_file_a()) << 9)  & their_pieces; // FIX: pre-shift mask
            for to in bb_iter(push1) { add_pawn_move(to - 10, to, us, &mut moves); }
            for to in bb_iter(push2) { moves.push((to - 20, to, None)); }
            // FIX: split loops — OR combination loses moves when two pawns attack the same square
            for to in bb_iter(cap_r) { add_pawn_move(to - 11, to, us, &mut moves); }
            for to in bb_iter(cap_l) { add_pawn_move(to - 9, to, us, &mut moves); }
            if let Some(ep) = self.ep_square {
                let attackers = ((1u128 << ep >> 11) & not_file_j() | (1u128 << ep >> 9) & not_file_a()) & pawns;
                for from in bb_iter(attackers) { moves.push((from, ep as u32, None)); }
            }
        } else {
            let push1 = (pawns >> 10) & empty;
            let push2 = ((pawns & rank_mask(6)) >> 10 & empty) >> 10 & empty;
            let cap_r = ((pawns & not_file_j()) >> 9)  & their_pieces; // FIX: pre-shift mask
            let cap_l = ((pawns & not_file_a()) >> 11) & their_pieces; // FIX: pre-shift mask
            for to in bb_iter(push1) { add_pawn_move(to + 10, to, us, &mut moves); }
            for to in bb_iter(push2) { moves.push((to + 20, to, None)); }
            // FIX: split loops — OR combination loses moves when two pawns attack the same square
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
        // Castling in Capablanca Chess (10×8):
        //   Kingside: king f(5)→i(8) +3 squares, rook j(9)→h(7)
        //     - empty: g(6), h(7), i(8)
        //     - king doesn't pass through check: g(6), h(7), i(8)
        //   Queenside: king f(5)→c(2) -3 squares, rook a(0)→d(3)
        //     - empty: b(1), c(2), d(3), e(4)
        //     - king doesn't pass through check: e(4), d(3), c(2)
        let us = self.side; let occ = self.occupancy(); let opp_att = self.attacks_by(1 - us);
        let back_rank = if us == 0 { 0u32 } else { 7u32 };
        let b = back_rank * 10;
        let king_sq = b + 5;
        if self.pieces[us][KING] & (1u128 << king_sq) == 0 { return; }
        if opp_att & (1u128 << king_sq) != 0 { return; }

        // ── Kingside: king f(5)→i(8), rook j(9)→h(7) ───────────────────────
        if self.castling & (1 << (us * 2)) != 0 {
            let must_empty = (1u128 << (b+6)) | (1u128 << (b+7)) | (1u128 << (b+8));
            let king_path  = (1u128 << (b+6)) | (1u128 << (b+7)) | (1u128 << (b+8));
            let rook_ok = self.pieces[us][ROOK] & (1u128 << (b+9)) != 0;
            if occ & must_empty == 0 && rook_ok && opp_att & king_path == 0 {
                moves.push((king_sq, b + 8, None));
            }
        }

        // ── Queenside: king f(5)→c(2), rook a(0)→d(3) ──────────────────────
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
        // FIX: compute capture BEFORE clearing pieces[them], otherwise the info is already lost
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
                // Kingside: king f(5)→i(8), rook j(9)→h(7)
                if to == back + 8 {
                    self.pieces[us][ROOK] &= !(1u128 << (back + 9));
                    self.pieces[us][ROOK] |=   1u128 << (back + 7);
                }
                // Queenside: king f(5)→c(2), rook a(0)→d(3)
                else if to == back + 2 {
                    self.pieces[us][ROOK] &= !(1u128 << (back + 0));
                    self.pieces[us][ROOK] |=   1u128 << (back + 3);
                }
            }
            self.castling &= !(3 << (us * 2));
        }
        // FIX: castling bits must match gen_castling:
        //   bit (us*2)   = kingside  → rook at sq 9 (white) / 79 (black)
        //   bit (us*2+1) = queenside → rook at sq 0 (white) / 70 (black)
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

    /// Checks for insufficient mating material.
    ///
    /// In Capablanca Chess a draw by material occurs when neither side
    /// can deliver checkmate even with the worst play from the opponent.
    ///
    /// Pieces that can ALWAYS deliver mate (not a draw):
    ///   Pawn, Rook, Queen, Archbishop (Bishop+Knight), Chancellor (Rook+Knight)
    ///
    /// Insufficient material cases:
    ///   K vs K
    ///   K+N vs K
    ///   K+B vs K
    ///   K+N vs K+N
    ///   K+N vs K+B
    ///   K+B vs K+B  (any colors — one bishop each)
    ///   K+N+N vs K  (two knights cannot force checkmate)
    ///
    /// NOT a draw (mate is possible):
    ///   K+Archbishop vs K  — arch combines bishop and knight moves, mate is real
    ///   K+Chancellor vs K  — trivial mate
    ///   K+Rook vs K        — trivial mate
    ///   K+Queen vs K       — trivial mate
    ///   Any pawn           — can promote
    fn is_insufficient_material(&self) -> bool {
        for c in 0..2 {
            // If there are pawns, rooks, queens, archbishops or chancellors — mate is possible
            if self.pieces[c][PAWN]  != 0 { return false; }
            if self.pieces[c][ROOK]  != 0 { return false; }
            if self.pieces[c][QUEEN] != 0 { return false; }
            if self.pieces[c][ARCH]  != 0 { return false; } // Archbishop can deliver mate
            if self.pieces[c][CHANC] != 0 { return false; } // Chancellor can deliver mate
        }

        // Only kings + minor pieces (bishops and knights)
        let w_knights = self.pieces[0][KNIGHT].count_ones();
        let w_bishops = self.pieces[0][BISHOP].count_ones();
        let b_knights = self.pieces[1][KNIGHT].count_ones();
        let b_bishops = self.pieces[1][BISHOP].count_ones();
        let w_minor = w_knights + w_bishops;
        let b_minor = b_knights + b_bishops;

        match (w_minor, b_minor) {
            // K vs K
            (0, 0) => true,
            // K+1 vs K  or  K vs K+1
            (1, 0) | (0, 1) => true,
            // K+N+N vs K — two knights cannot force mate without assistance
            (2, 0) if w_knights == 2 => true,
            (0, 2) if b_knights == 2 => true,
            // K+1 vs K+1 — bishops and knights can't mate each other
            (1, 1) => true,
            // Everything else — mate is theoretically possible
            _ => false,
        }
    }

    fn material_balance(&self) -> i32 {
        // Standard weights + Capablanca pieces
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

    /// Converts a move to a policy-vector index in CANONICAL coordinates.
    /// When side=1 from/to are vertically flipped before encoding.
    /// This guarantees the network's policy uses one coordinate system
    /// regardless of side.
    fn move_to_idx(from: u32, to: u32, promo: Option<usize>, side: usize) -> usize {
        let (f, t) = if side == 1 { (flip_sq(from), flip_sq(to)) } else { (from, to) };
        match promo {
            None => (f * 80 + t) as usize,
            Some(p) => {
                let pi = match p { QUEEN => 0, ROOK => 1, BISHOP => 2, KNIGHT => 3, ARCH => 4, CHANC => 5, _ => 0 };
                // Include file_from to distinguish two pawns on adjacent files
                // that can both move/capture to the same promotion square.
                // base: 0..=99 (10 × 10 files).
                // Max index: 6400 + 99*6 + 5 = 6999 → POLICY_SIZE = 7000.
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

/// Vertical flip of a square: rank r → rank (7-r), file preserved.
/// For canonical input form: black moves are converted to a position "as if white were moving",
/// so the network always sees the board from the side to move (LC0 board.Mirror()).
#[inline]
fn flip_sq(sq: u32) -> u32 { (7 - sq / 10) * 10 + (sq % 10) }

/// Flip all bits in a bitboard.
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
// The network sees not only the current position but also 7 previous → 8 total.
// This provides a critical signal: "what just moved" (attacking piece,
// whether defense is under threat), repetitions visible directly (no special plan),
// overall positional dynamics.
//
// Plane layout (total 139 = 8*17 + 3):
//   per history slot h ∈ 0..8 (newest=0):
//     h*17 + 0..7   OUR pieces (P,N,B,R,Q,A,C,K) [canonical-flipped if side=1]
//     h*17 + 8..15  THEIR pieces
//     h*17 + 16     repetition flag (1.0 if this position is a repetition)
//   136  castling (4 zones × 20 squares) — current position only
//   137  halfmove / 100
//   138  all-ones (LC0 edge-detection)
pub const HISTORY_LEN: usize = 8;
pub const PLANES_PER_BOARD: usize = 17;  // 8 ours + 8 theirs + 1 rep
pub const META_PLANES: usize = 3;        // castling + halfmove + ones
pub const TOTAL_INPUT_PLANES: usize = HISTORY_LEN * PLANES_PER_BOARD + META_PLANES;  // 139

/// Encode position history into a canonical plane for the NN.
/// history[0] = current position (most recent); history[i] = i moves ago.
/// Empty slots (if history is short) — filled with zeros.
/// `current_side` determines the canonical flip of ALL positions (flip relative to
/// the side moving in the CURRENT position, so "ours" is always at the bottom).
/// `rep_flags[i]` = whether position i is a repetition.
fn boards_to_tensor(history: &[Board], rep_flags: &[bool],
                    current_side: usize, halfmove: u32, castling: u8) -> Vec<f32> {
    let mut t = vec![0.0f32; TOTAL_INPUT_PLANES * 80];
    let do_flip = current_side == 1;

    for h in 0..HISTORY_LEN {
        if h >= history.len() { break; }
        let board = &history[h];
        // Canonical our/their mapping: "ours" in the plane is the side
        // moving in the CURRENT (newest) position. For historical positions
        // (where the state may differ) we use the same definition.
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

    // Meta: castling, halfmove, all-ones — current position only
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

// FIX: cache legal moves — for one position they are computed only once.
// In generate_games each move called gen_legal() three times:
//   is_game_over() → gen_legal()
//   get_legal_moves_int() → gen_legal()
//   (inside MCTS copy + expand) → gen_legal()
// Cache is reset only on make_move_int().
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct CapablancaEngine {
    board: Board,
    legal_cache: Option<Vec<(u32, u32, Option<usize>)>>,
    position_history: Vec<u64>,
    // board_history: snapshot of the last HISTORY_LEN boards INCLUDING the current one.
    // board_history[0] = current position, board_history[1] = one move back, etc.
    // Used for history planes (LC0-style).
    board_history: Vec<Board>,
}

impl CapablancaEngine {
    fn ensure_legal_cache(&mut self) {
        if self.legal_cache.is_none() {
            self.legal_cache = Some(self.board.gen_legal());
        }
    }

    /// Returns true if the position in board_history[i] is a repetition
    /// (hash appears at least once more in position_history up to that point).
    /// For history planes — gives the network a repetition signal per slot.
    fn rep_flags(&self) -> Vec<bool> {
        self.board_history.iter().map(|b| {
            let h = compute_board_hash(b);
            // How many times this position appeared in total in game_history.
            // If > 1 — repetition. (Current position is also in position_history.)
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

    /// List of pieces on the board in RAW (non-canonical) coordinates.
    /// Returns Vec<(color, piece_type, square)>:
    ///   color: 0=white, 1=black
    ///   piece_type: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, ARCH=5, CHANC=6, KING=7
    ///   square: 0..80, square = rank*10 + file
    /// Used by the GUI for rendering — needs RAW view, not canonical flipped form.
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
        // History planes (LC0-style): network sees 8 most recent positions (newest first).
        // Current = board_history[0], previous = board_history[1], etc.
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
        self.legal_cache = None; // reset cache after move
        // Maintain position history for 3-fold repetition.
        // On irreversible move (capture/pawn) halfmove_clock=0 → old positions can't repeat.
        if self.board.halfmove_clock == 0 {
            self.position_history.clear();
        }
        self.position_history.push(compute_board_hash(&self.board));
        // board_history: push NEW current position to front, trim to HISTORY_LEN.
        // board_history[0] = current, [1] = previous, ...
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
        // Canonical index: for black moves from/to are flipped.
        Some(Board::move_to_idx(f, t, p, self.board.side))
    }

    pub fn is_game_over(&mut self) -> bool {
        if self.board.halfmove_clock >= 100 { return true; }
        if self.board.is_insufficient_material() { return true; }
        // 3-fold: current position already in history (pushed in make_move_int + new()).
        let cur = compute_board_hash(&self.board);
        let repeats = self.position_history.iter().filter(|&&h| h == cur).count();
        if repeats >= 3 { return true; }
        self.ensure_legal_cache();
        self.legal_cache.as_ref().unwrap().is_empty()
    }

    pub fn game_result(&mut self) -> f32 {
        if self.board.halfmove_clock >= 100 { return 0.0; }
        if self.board.is_insufficient_material() { return 0.0; }
        // 3-fold repetition = draw
        let cur = compute_board_hash(&self.board);
        let repeats = self.position_history.iter().filter(|&&h| h == cur).count();
        if repeats >= 3 { return 0.0; }
        self.ensure_legal_cache();
        if self.legal_cache.as_ref().unwrap().is_empty() {
            if self.board.in_check(self.board.side) {
                return if self.board.side == 0 { -1.0 } else { 1.0 };
            }
            return 0.0; // Stalemate
        }
        0.0
    }

    /// Early adjudication result by material.
    ///
    /// Returns:
    ///   ±1.0 — decisive advantage (≥8 points): one side has a major piece
    ///          the opponent lacks. Counted as a win.
    ///   ±0.5 — moderate advantage (3-7 points): likely win, but not guaranteed.
    ///    0.0 — approximate equality (< 3 points).
    ///
    /// 8-point threshold chosen for Capablanca: archbishop worth ~8, chancellor ~10.
    /// If one side has arch/chanc/queen and the other doesn't — advantage is decisive.
    pub fn material_result(&self) -> f32 {
        let balance = self.board.material_balance();
        if balance >= 8  { return  1.0; }
        if balance <= -8 { return -1.0; }
        if balance > 3  { return  0.5; }
        if balance < -3 { return -0.5; }
        0.0
    }

    pub fn adjudication_result(&self) -> Option<f32> { None }

    /// 64-bit hash of the current position: pieces + side + castling + en-passant.
    /// Excludes the halfmove clock, so it is a pure transposition key — two
    /// positions reached by different move orders share the same hash.
    pub fn position_hash(&self) -> u64 {
        compute_board_hash(&self.board)
    }
}

// MODULE REGISTRATION WITH EXPLICIT NAME
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
        // Any black move in canonical coordinates must have
        // THE SAME index as the symmetric white move.
        // Example: white e2e4 = black e7e5 in canonical form.
        let promos = [QUEEN, ROOK, BISHOP, KNIGHT, ARCH, CHANC];
        for from_file in 0u32..10 {
            for to_file in 0u32..10 {
                for from_rank in 0u32..8 {
                    for to_rank in 0u32..8 {
                        let from_w = from_rank * 10 + from_file;
                        let to_w   = to_rank   * 10 + to_file;
                        // Symmetric move for black: same move on the flipped board
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
        // Canonical form: starting position for white and black must match
        // in the first 8 piece-planes (OUR pieces always on bottom rank).
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
        // Tensor size = TOTAL_INPUT_PLANES * 80
        assert_eq!(t_w.len(), TOTAL_INPUT_PLANES * 80);
    }

    #[test]
    fn test_history_planes_layout() {
        // 8 history slots + 3 meta = 139 planes (for 17 planes per board)
        assert_eq!(TOTAL_INPUT_PLANES, HISTORY_LEN * PLANES_PER_BOARD + META_PLANES);
        assert_eq!(TOTAL_INPUT_PLANES, 139);
    }

    #[test]
    fn test_root_is_not_repetition() {
        // CRITICAL regression test:
        // On first root expand — position_history contains root_hash (single entry).
        // rep_count_at_leaf(root, root_board) MUST return 1 (history only),
        // not 2 — otherwise root is marked as 2-fold terminal → MCTS doesn't work.
        let mcts = SingleMcts::new(Board::start());
        let root = mcts.root;
        let count = mcts.rep_count_at_leaf(root, &mcts.root_board);
        assert_eq!(count, 1, "root should not be counted as repetition on first expand");
        assert!(!mcts.is_repetition_at_leaf(root, &mcts.root_board),
                "root should not be a 2-fold terminal");
    }

    #[test]
    fn test_startpos_legal_moves() {
        let mut board = Board::start();
        let legal = board.gen_legal();
        // In Capablanca starting position: 10 pawn moves + 4 knight moves + 4 archbishop/chancellor moves
        // Exact count depends on rules, but should be > 20
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
        // Helper function: creates empty board with kings only
        fn kings_only() -> Board {
            let mut b = Board { pieces: [[0; 8]; 2], side: 0, castling: 0,
                ep_square: None, halfmove_clock: 0, fullmove: 1 };
                b.pieces[0][KING] = 1u128 << 5;   // white king f1
                b.pieces[1][KING] = 1u128 << 75;  // black king f8
                b
        }

        // K vs K — draw
        let b = kings_only();
        assert!(b.is_insufficient_material(), "K vs K must be draw");

        // K+Knight vs K — draw
        let mut b = kings_only();
        b.pieces[0][KNIGHT] = 1u128 << 1;
        assert!(b.is_insufficient_material(), "K+N vs K must be draw");

        // K+Bishop vs K — draw
        let mut b = kings_only();
        b.pieces[0][BISHOP] = 1u128 << 3;
        assert!(b.is_insufficient_material(), "K+B vs K must be draw");

        // K+2 Knights vs K — draw (forced mate impossible)
        let mut b = kings_only();
        b.pieces[0][KNIGHT] = (1u128 << 1) | (1u128 << 8);
        assert!(b.is_insufficient_material(), "K+N+N vs K must be draw");

        // K+1 vs K+1 — draw
        let mut b = kings_only();
        b.pieces[0][KNIGHT] = 1u128 << 1;
        b.pieces[1][BISHOP] = 1u128 << 66;
        assert!(b.is_insufficient_material(), "K+N vs K+B must be draw");

        // K+Archbishop vs K — NOT a draw
        let mut b = kings_only();
        b.pieces[0][ARCH] = 1u128 << 2;
        assert!(!b.is_insufficient_material(), "K+A vs K must NOT be draw");

        // K+Chancellor vs K — NOT a draw
        let mut b = kings_only();
        b.pieces[0][CHANC] = 1u128 << 7;
        assert!(!b.is_insufficient_material(), "K+C vs K must NOT be draw");

        // K+Rook vs K — NOT a draw
        let mut b = kings_only();
        b.pieces[0][ROOK] = 1u128 << 0;
        assert!(!b.is_insufficient_material(), "K+R vs K must NOT be draw");

        // K+Pawn vs K — NOT a draw
        let mut b = kings_only();
        b.pieces[0][PAWN] = 1u128 << 10;
        assert!(!b.is_insufficient_material(), "K+P vs K must NOT be draw");

        println!("✅ insufficient material: all cases correct");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUST MCTS — tree lives in Rust, Python only feeds the neural network
// ═══════════════════════════════════════════════════════════════════════════════

const POLICY_SIZE_MCTS: usize = 7000;
const VIRTUAL_LOSS_V: i32 = 3;
// MCTS parameters from lc0 params.cc — tuned over millions of games.
const C_PUCT_V: f32 = 1.745;        // CPuct (lc0)
const C_PUCT_FACTOR: f32 = 3.894;   // CPuctFactor — multiplier for logarithmic growth
const C_PUCT_BASE: f32 = 38739.0;   // CPuctBase — inflection point
// Dirichlet alpha computed dynamically: max(10/n_children, 0.1). See expand().
const DIRICHLET_EPS_V: f64 = 0.35; // raised from 0.25 — more exploration at start

// MLH normalization (must match CapablancaNet.MLH_PLY_NORM in Python).
// NN outputs [0,1] = "fraction of MLH_PLY_NORM half-moves". Multiplied in backup.
const MLH_PLY_NORM: f32 = 200.0;
// LC0 MEvaluator constants (params.cc:597-600). Effect activates only when |q| > THRESHOLD.
const M_SLOPE: f32 = 0.0027;
const M_CAP: f32 = 0.0345;
const M_THRESHOLD: f32 = 0.8;

// MctsNode without Board — like lc0.
// Board is stored only in SingleMcts.root_board.
// Position is reconstructed in expand/collect_leaves by traversing from root.
// Size: ~62 bytes instead of ~300 bytes → 8192 nodes = 500KB (fits in L2).
//
// Bounds (lower, upper): proven value range of a node from its POV.
//   -1 = LOSS, 0 = DRAW, +1 = WIN. Default (-1, +1) = nothing proven.
//   Terminals: lower == upper == game result.
//   Bounds prop. (LC0 StickyEndgames, search.cc:2302) — when all children lose →
//   parent is proven as win. Used in select() to redirect simulations from solved branches.
// Terminal kinds — for correct tree reuse.
// Natural terminals (checkmate/stalemate/50-move/insufficient) — PERMANENT (position-derived).
// TwoFold — path-dependent: may become invalid after root shift.
// Default (0) = either non-terminal, or bounds-proven (also path-dependent via children).
const TERMINAL_KIND_NONE: u8 = 0;     // not a terminal OR bounds-proven
const TERMINAL_KIND_NATURAL: u8 = 1;  // checkmate/stalemate/50-move/insufficient — permanent
const TERMINAL_KIND_TWOFOLD: u8 = 2;  // 2-fold inside tree — path-dependent

struct MctsNode {
    move_from_parent: u32,  // move used to reach this node
    prior: f32,
    visits: i32,
    wl: f32,               // running average Q (lc0: FinalizeScoreUpdate)
    d: f32,                // running average Draw probability (for contempt: Q = WL + draw_score*D)
    m: f32,                // running average remaining plies (LC0 MLH). In PLY units (not normalized).
    virtual_loss: i32,
    children: Vec<usize>,
    is_expanded: bool,
    is_terminal: bool,
    terminal_kind: u8,     // see TERMINAL_KIND_*
    lower: i8,             // lower bound (proven worst-case) from POV of this node
    upper: i8,             // upper bound (proven best-case)
    side: u8,              // whose turn to move in this node (0=white, 1=black)
    parent: Option<usize>,
    // Position hash for O(1) repetition lookup. Set in expand().
    // 0 for unexpanded nodes — invalid for comparison.
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
            lower: -1, upper: 1,  // nothing proven
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

    // Q biased by a contempt factor on the running draw probability.
    // Q_eff = wl + c * d (per-visit averages). vloss correction matches q():
    // visits scale up the biased Q, then we subtract vloss and divide by (visits + vloss).
    fn q_with_contempt(&self, contempt: f32) -> f32 {
        if contempt == 0.0 { return self.q(); }
        let biased = self.wl + contempt * self.d;
        if self.virtual_loss > 0 {
            let total = self.visits + self.virtual_loss;
            if total > 0 { (biased * self.visits as f32 - self.virtual_loss as f32) / total as f32 }
            else { 0.0 }
        } else {
            biased
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

// FIX: original Marsaglia-Tsang only works for alpha >= 1/3.
// For alpha=0.3: d = 0.3 - 1/3 = -0.033 → sqrt(9*d) = sqrt(-0.3) = NaN.
// NaN priors → select() always chose children[0] → entropy=0, top1=1.
// Fix: sample_gamma with reduction Gamma(a) = Gamma(a+1) * U^(1/a) for a < 1.
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
        // Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha), correct for any alpha > 0
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
    // root_history[0] = root_board, [1..] = moves BACKWARD from root (up to HISTORY_LEN-1 entries).
    // During leaf-walk we collect history by prepending new positions.
    root_history: Vec<Board>,
    // Reusable buffer for collect_leaves — no allocations per step
    leaf_tensor_buf: Vec<f32>,
    // KLD-early-exit: snapshot of previous root visits distribution.
    // Vec<(move_int, visits)>. None if no snapshot yet (or after root reset).
    // Computed in Rust to avoid marshalling 7000-vector to Python every 2 MCTS steps.
    kld_prev_snapshot: Option<Vec<(u32, i32)>>,
    // Accumulating tree reuse: root visits AT THE START of the current move
    // (i.e. visits inherited via tree reuse from the previous move).
    // sims_done_this_move = root.visits - move_start_visits → honest remaining
    // new budget for best_move_is_decided, without "choking" due to reused visits.
    move_start_visits: i32,
    // Contempt: PUCT selection uses Q_eff = wl + contempt * d (per node, child POV).
    // 0.0 = standard MCTS, > 0 = avoid draws (aggressive play vs weaker opponents),
    // < 0 = accept draws. Applied only inside `select`, never to backup/bounds —
    // statistics remain valid if contempt changes between moves.
    contempt: f32,
    // Dirichlet noise on root priors (Lc0 self-play exploration). True for
    // training; flipped off for eval / FSF / lagged play where we want a
    // deterministic measurement of the network's actual choice.
    add_dirichlet: bool,
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
        arena.get_mut(root).position_hash = initial_hash;  // root always has valid hash
        let buf_cap = 8192 * TOTAL_INPUT_PLANES * 80;
        SingleMcts {
            arena, root, root_board: board,
            pending: Vec::new(), pending_boards: Vec::new(),
            position_history: vec![initial_hash],
            root_history,
            leaf_tensor_buf: Vec::with_capacity(buf_cap),
            kld_prev_snapshot: None,
            move_start_visits: 0,
            contempt: 0.0,
            add_dirichlet: true,
        }
    }

    /// Snapshot of current root visits distribution for KLD-early-exit.
    /// Stores (move_int, visits) for each root child.
    fn kld_take_snapshot(&mut self) {
        let root_children = self.arena.get(self.root).children.clone();
        let mut snap: Vec<(u32, i32)> = Vec::with_capacity(root_children.len());
        for ci in root_children {
            let c = self.arena.get(ci);
            snap.push((c.move_from_parent, c.visits));
        }
        self.kld_prev_snapshot = Some(snap);
    }

    /// KL(prev || curr) over root visits. Returns +inf if no snapshot
    /// or if either side has 0 visits (KL undefined).
    /// O(N_children^2) due to linear search, but N is usually < 60.
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
            // Find curr visits for move m (linear search; N_children usually < 60)
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

    /// Resets the KLD snapshot. Called on make_move (root shift),
    /// renoise_root, or any change to root structure.
    fn kld_reset(&mut self) {
        self.kld_prev_snapshot = None;
    }

    // Reconstructs Board for node idx by traversing from root.
    // O(depth) — usually 10-30 moves, fast.
    fn board_at(&self, idx: usize) -> Board {
        // Collect path from node up to root
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
        // Apply moves from root downward
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

    /// Reconstructs Board AND history (HISTORY_LEN last positions) for node idx.
    /// history[0] = leaf, history[1] = 1 move back, ..., history[k] = root_history[k - path_len].
    /// Used for history-planes encoding of a leaf.
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
        // Replay forward, recording intermediate positions
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

        // Collect history newest→oldest:
        //   first the tree path (from leaf toward root, excluding duplicate root_board)
        //   then root_history[1..] (root_board already taken as last path element)
        let mut history: Vec<Board> = Vec::with_capacity(HISTORY_LEN);
        for b in forward_boards.iter().rev() {
            history.push(b.clone());
            if history.len() >= HISTORY_LEN { break; }
        }
        if history.len() < HISTORY_LEN {
            // root_history[0] == self.root_board → skip, start from [1]
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
            // Terminals: back up known value upward (LC0-style).
            // Without this the simulation is lost → parent Q doesn't converge to terminal,
            // PUCT keeps experimenting instead of using the known outcome.
            let (is_terminal, terminal_v, terminal_d, terminal_m) = {
                let n = self.arena.get(idx);
                (n.is_terminal, n.wl, n.d, n.m)
            };
            if is_terminal {
                // Terminal: m = 0 (game over), plies_from_leaf incremented in backup
                self.backup(idx, terminal_v, terminal_d, terminal_m);
                return None;
            }
            let node = self.arena.get(idx);
            if !node.is_expanded { return Some(idx); }
            if node.children.is_empty() { return None; }

            let parent_visits = (self.arena.get(idx).visits + self.arena.get(idx).virtual_loss).max(1);
            let sqrt_n = (parent_visits as f32).sqrt();

            // Dynamic CPUCT using lc0 formula (params.cc):
            //   cpuct = CPuct + CPuctFactor * ln((N + CPuctBase) / CPuctBase)
            // Tuned values: CPuct=1.745, CPuctFactor=3.894, CPuctBase=38739
            // At N=0: cpuct ≈ 1.745, at N=10K: ≈ 2.7, at N=100K: ≈ 4.0
            let cpuct = C_PUCT_V
                + C_PUCT_FACTOR * ((parent_visits as f32 + C_PUCT_BASE) / C_PUCT_BASE).ln();

            // FPU strategies (LC0 params.cc:567-572):
            //   non-root: "reduction" — fpu = parent_q - 0.330 * sqrt(visited_policy_sum)
            //   root:     "absolute" — fpu = +1.0 (give maximum exploration to root children)
            // At root unvisited moves get maximally optimistic score → quickly
            // scan the ENTIRE root move set before going deeper into any one of them.
            // Important for Dirichlet noise: otherwise noise only applies when children are visited,
            // but at start all children are unvisited and get the same fpu=parent_q.
            let parent_q = self.arena.get(idx).q();
            let n_ch = self.arena.get(idx).children.len();
            let is_root = idx == self.root;
            let fpu = if is_root {
                1.0f32  // absolute strategy: forced exploration at root
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
            // Pre-compute data for M-utility (LC0 MEvaluator, search.cc:111).
            // Effect activates only if parent_q is "strongly decided" (|q| > THRESHOLD).
            // This makes M-bias cheap in normal positions and strong in endgames.
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
                // Contempt biases Q toward/away from draws — 0.0 means standard MCTS.
                let q_val = if started > 0 { -c.q_with_contempt(self.contempt) } else { fpu };
                let mut score = q_val + cpuct * c.prior * sqrt_n / (1 + started) as f32;
                // Bounds-aware bias (LC0 StickyEndgames):
                //   c.upper == -1 → child guaranteed losing → parent wins: boost.
                //   c.lower == +1 → child guaranteed winning → parent loses: suppress.
                if c.upper == -1 { score += 100.0; }
                else if c.lower == 1 { score -= 100.0; }
                // M-utility: prefer short wins / long losses.
                // Applied only for visited children (for unvisited m=0, no data).
                if m_active && started > 0 {
                    let dm = (M_SLOPE * (c.m - parent_m)).clamp(-M_CAP, M_CAP);
                    // q_val = -c.q() — move evaluation from parent's perspective.
                    //   q_val > 0 (winning move for parent) → sign(-q_val) = -1 → penalize dm>0 (long win)
                    //   q_val < 0 (losing move) → sign(-q_val) = +1 → bonus dm>0 (delay the loss)
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

    /// Returns total_count = how many times the leaf position has occurred in total:
    ///   - in the actual game history (position_history includes root)
    ///   - in the path from ROOT (excl.) to leaf (incl., if leaf != root)
    ///
    /// Optimization: uses position_hash stored in nodes during expand.
    /// Old version replayed moves forward from root = O(P²) with board.apply_move.
    /// New version = O(P) reads from arena.
    fn rep_count_at_leaf(&self, leaf_idx: usize, leaf_board: &Board) -> u32 {
        let leaf_hash = compute_board_hash(leaf_board);
        let history_count = self.position_history.iter()
            .filter(|&&h| h == leaf_hash).count() as u32;

        // Path count: count leaf only if it is NOT root (root already in position_history,
        // don't double-count). Then walk UP: count intermediate nodes (not root).
        //
        // Without the `leaf_idx == self.root` check, start=1 on first root expand gave
        // total = history(1) + path(1) = 2 → root marked as 2-fold terminal
        // → MCTS never expanded children → entire search broke.
        let mut path_count = if leaf_idx == self.root { 0u32 } else { 1u32 };
        let mut cur = leaf_idx;
        while let Some(parent) = self.arena.get(cur).parent {
            if parent == self.root { break; }
            let pn = self.arena.get(parent);
            // position_hash is set on expand. If 0 → unexpanded, skip comparison.
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

    // board is passed externally (already reconstructed via board_at or stored in pending_boards)
    fn expand(&mut self, idx: usize, board: &Board, policy: &[f32], add_noise: bool, rng: &mut u64) {
        // First record position_hash — needed for O(1) repetition check in descendants.
        let board_hash = compute_board_hash(board);
        self.arena.get_mut(idx).position_hash = board_hash;

        let legal = board.gen_legal();
        // Terminals: natural (checkmate/stalemate/50-move/insufficient) — permanent,
        // 2-fold — path-dependent (LC0-style), requires revalidation after tree reuse.
        let natural_terminal = legal.is_empty()
            || board.halfmove_clock >= 100
            || board.is_insufficient_material();
        let twofold_terminal = !natural_terminal && self.is_repetition_at_leaf(idx, board);

        if natural_terminal || twofold_terminal {
            // Determine result from POV of the side at this node
            let side = board.side;
            let v: i8 = if legal.is_empty() {
                if board.in_check(side) { -1 } else { 0 }  // mate vs stalemate
            } else { 0 };  // draw rule
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
        // Child nodes: store only the move and prior — Board not needed
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
        //   wl += (v*sign - wl) / (n + 1)            — sign alternates (negamax)
        //   d  += (d_leaf - d) / (n + 1)             — D doesn't flip (draw is symmetric)
        //   m  += (mlh_leaf + plies_from_leaf - m) / (n + 1)
        //     ↑ CRITICAL FIX (Gemini): add 1 at EVERY level going up from leaf.
        //     Leaf — m_ply from leaf. Parent — m_ply+1 (one move earlier = one more move left).
        //     Grandparent — m_ply+2. And so on.
        // virtual_loss is NOT touched here — it is owned by the caller, which
        // pairs every collect-time `apply_vloss(+V)` with an explicit
        // `apply_vloss(-V)` in apply_inference. Decrementing here too caused
        // a double-removal: each finished leaf zeroed vloss along its path
        // including the contributions of *other* in-flight leaves through
        // shared ancestors. That broke virtual-loss-as-divergence: parallel
        // selects collapsed back onto the same line. Same issue for the
        // terminal case in `select`, which calls backup directly without a
        // prior apply_vloss(+V) — now it cannot steal anyone's vloss.
        let mut sign = 1.0f32;
        let mut plies_from_leaf = 0.0f32;
        loop {
            let n = self.arena.get_mut(idx);
            n.visits += 1;
            n.wl += (value * sign - n.wl) / n.visits as f32;
            n.d  += (draw_prob - n.d) / n.visits as f32;
            n.m  += (mlh_ply + plies_from_leaf - n.m) / n.visits as f32;
            sign *= -1.0;
            plies_from_leaf += 1.0;
            match n.parent { Some(p) => idx = p, None => break }
        }
    }

    /// Bounds propagation from a newly proven terminal upward (LC0 StickyEndgames).
    /// When a node becomes terminal or gets tighter bounds — parent may also be proven.
    /// new_lower_parent = max over children of -child.upper  (best guaranteed result)
    /// new_upper_parent = max over children of -child.lower  (best possible result)
    /// If new_lower == new_upper → parent proven → mark as terminal and continue upward.
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
            // Previous p.wl was a running-average from NN evaluations. Now this value
            // is proven exactly as `new_wl`. All ancestors averaged this node over
            // p_visits visits with stale old_wl. Apply correction upward.
            // Formula: ancestor.wl += sign * p_visits * (new_wl - old_wl) / ancestor.visits
            // Sign alternates (negamax). D does not flip sign.
            if p_visits > 0 {
                let v_delta = (new_wl - old_wl) * p_visits as f32;
                let d_delta = (new_d  - old_d ) * p_visits as f32;
                let mut sign = -1.0f32;  // first ancestor in flipped POV
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
        true  // bounds shifted — continue propagation upward
    }

    // Early stopping — if the best move mathematically cannot be caught by
    // the second-best within remaining simulations, skip this game.
    // IMPORTANT: only triggers if root is already expanded AND has enough visits.
    // Without the root.visits check, tree was skipped BEFORE first simulation → zero policy.
    fn best_move_is_decided(&self, sims_remaining: i32) -> bool {
        let root = self.arena.get(self.root);
        // Don't touch unexpanded trees or trees with fewer than 2 children
        if !root.is_expanded || root.children.len() < 2 { return false; }
        // Need a minimum visit threshold for statistics to be meaningful
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

    // Version with reusable buffer — no Vec<Vec<f32>> allocations
    fn collect_leaves_into_buf(&mut self, parallel: usize, _rng: &mut u64) -> usize {
        self.pending.clear();
        self.pending_boards.clear();
        self.leaf_tensor_buf.clear();
        let mut count = 0usize;
        for _ in 0..parallel {
            if let Some(leaf) = self.select() {
                if self.arena.get(leaf).is_terminal { continue; }
                // Duplicate protection: select() may return the same unexpanded node
                // repeatedly (vloss doesn't help for unexpanded — we never reach the PUCT loop).
                // Without this check, visits inflate and GPU evaluates identical positions.
                if self.pending.contains(&leaf) { continue; }
                self.apply_vloss(leaf, VIRTUAL_LOSS_V);
                let (leaf_board, history) = self.board_with_history_at(leaf);
                // Rep flag only for the current (leaf) position — most important information.
                // Leave old slots at 0 for simplicity (LC0 V2 also often skips them).
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

    // Old version — kept for compatibility
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
            let is_root = leaf == self.root && self.add_dirichlet;
            let was_unexpanded = !self.arena.get(leaf).is_expanded;
            if was_unexpanded {
                if let Some(board) = boards.get(i) {
                    self.expand(leaf, board, &policies[i], is_root, rng);
                }
            }
            // Terminal: expand already set wl/d/bounds. m_terminal = 0 (game already over).
            // Non-terminal: use NN value/draw/mlh. mlh from net ∈ [0,1] → multiply by NORM=200 → PLY.
            let leaf_terminal = self.arena.get(leaf).is_terminal;
            let (v, d, m_ply) = if leaf_terminal {
                let n = self.arena.get(leaf);
                (n.wl, n.d, 0.0_f32)  // terminal = 0 plies remaining
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

    // Version without Vec<Vec<f32>> — accepts flat policy slice
    fn apply_inference_flat(&mut self, pol_flat: &[f32], policy_size: usize,
                            values: &[f32], draws: &[f32], mlhs: &[f32], rng: &mut u64) {
        let pending = std::mem::take(&mut self.pending);
        let boards  = std::mem::take(&mut self.pending_boards);
        for (i, leaf) in pending.into_iter().enumerate() {
            if i >= values.len() { break; }
            self.apply_vloss(leaf, -VIRTUAL_LOSS_V);
            let is_root = leaf == self.root && self.add_dirichlet;
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
                                        // Canonical index (see Board::move_to_idx).
                                        let idx = Board::move_to_idx(f, t, p, side);
                                        if idx < POLICY_SIZE_MCTS { pol[idx] = c.visits as f32 / total as f32; }
                                    }
                                }
                                pol
                            }

                            fn is_over(&mut self) -> bool {
                                if self.root_board.halfmove_clock >= 100 { return true; }
                                if self.root_board.is_insufficient_material() { return true; }
                                // Three-fold repetition: current position already in history (pushed in make_move),
                                // 3-fold = this position occurred >= 3 times.
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

                            // Recursively copies a subtree from self.arena into new_arena (without Board).
                            fn copy_subtree(&self, old_idx: usize, new_arena: &mut Arena, new_parent: Option<usize>) -> usize {
                                let old_node = self.arena.get(old_idx);
                                let mut new_node = MctsNode::new(
                                    old_node.move_from_parent,
                                    old_node.prior,
                                    old_node.side,
                                    new_parent,
                                );
                                // Preserve all accumulated statistics and caches.
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
                                // Apply move to root_board
                                let pv = m_int & 0b111;
                                let t  = (m_int >> 3) & 0x7F;
                                let f  = (m_int >> 10) & 0x7F;
                                let p  = if pv == 0 { None } else { Some((pv - 1) as usize) };
                                self.root_board.apply_move(f, t, p);
                                let new_side = self.root_board.side as u8;

                                let child_idx = self.arena.get(self.root).children.iter()
                                .copied()
                                .find(|&ci| self.arena.get(ci).move_from_parent == m_int);

                                // Tree GC — copy only the needed subtree into a new arena.
                                let mut new_arena = Arena::new(8192);
                                self.root = if let Some(ci) = child_idx {
                                    self.copy_subtree(ci, &mut new_arena, None)
                                } else {
                                    // Move was not in the tree — create root from scratch
                                    new_arena.add(MctsNode::new(m_int, 1.0, new_side, None))
                                };
                                self.arena = new_arena;
                                // Accumulating tree reuse: record inherited visits —
                                // new simulation budget will be counted on top of them.
                                self.move_start_visits = self.arena.get(self.root).visits;
                                self.pending.clear();
                                self.pending_boards.clear();
                                // Update history: on irreversible move (capture/pawn) old positions cannot repeat
                                let new_hash = Self::board_hash(&self.root_board);
                                if self.root_board.halfmove_clock == 0 {
                                    self.position_history.clear();
                                }
                                self.position_history.push(new_hash);
                                // board history (LC0 history planes): push new position to front.
                                self.root_history.insert(0, self.root_board.clone());
                                if self.root_history.len() > HISTORY_LEN {
                                    self.root_history.truncate(HISTORY_LEN);
                                }
                                // LC0 EnsureNodeTwoFoldCorrectForDepth (search.cc:1532):
                                // After root shift, revalidate path-dependent terminals.
                                self.revalidate_after_root_shift();
                            }

                            /// Revalidate terminals after tree reuse (LC0 EnsureNodeTwoFoldCorrect):
                            ///   - Natural terminals (checkmate/stalemate/50-move/insufficient) — permanent, keep.
                            ///   - 2-fold terminals — recompute rep_count for the new path from ROOT.
                            ///     If no longer a repetition → revert flag, re-expand.
                            ///   - Bounds-prop terminals (terminal_kind=NONE+is_terminal=true) —
                            ///     conservatively reset bounds (will be rediscovered by next simulations).
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
                                        // Reconstruct board + check for repetition
                                        let board = self.board_at(idx);
                                        if self.rep_count_at_leaf(idx, &board) >= 2 {
                                            continue;  // still a repetition
                                        }
                                        // No longer a repetition — reset as leaf for re-expand.
                                        let n = self.arena.get_mut(idx);
                                        n.is_terminal = false;
                                        n.is_expanded = false;
                                        n.terminal_kind = TERMINAL_KIND_NONE;
                                        n.lower = -1;
                                        n.upper = 1;
                                        n.children.clear();
                                        // wl/d/visits preserved — will be refined by new simulations.
                                    } else {
                                        // kind == NONE + is_terminal=true → bounds-prop terminal.
                                        // Children preserved, reset flag, may be rediscovered next simulation.
                                        let n = self.arena.get_mut(idx);
                                        n.is_terminal = false;
                                        n.lower = -1;
                                        n.upper = 1;
                                    }
                                }
                            }

                            /// Re-applies Dirichlet noise to priors of current root's children.
                            /// Called after make_move with tree reuse — otherwise exploratory
                            /// noise only applies on the FIRST move of the entire game.
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
                                // Without this, early-exit in select() operates on stale order.
                                let mut pairs: Vec<(usize, f32)> = children.iter()
                                    .map(|&ci| (ci, self.arena.get(ci).prior))
                                    .collect();
                                pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                self.arena.get_mut(root_idx).children = pairs.into_iter().map(|(ci, _)| ci).collect();
                            }
}

/// BatchState stores leaf_counts together with batch data.
/// This allows double-buffering to apply policies to the correct batch —
/// self.leaf_counts is overwritten by the next collect_leaves, but BatchState is not.
struct BatchState {
    leaf_counts: Vec<usize>,
}

/// RustMCTS — batched MCTS for N games simultaneously.
#[pyclass]
pub struct RustMCTS {
    games: Vec<SingleMcts>,
    parallel_sims: usize,
    rng: u64,
    leaf_game_map: Vec<usize>,
    leaf_counts: Vec<usize>,
    prev_batch: Option<BatchState>,  // for correct double-buffering
}

#[pymethods]
impl RustMCTS {
    #[new]
    pub fn new(engines: Vec<PyRef<CapablancaEngine>>, parallel_sims: usize) -> Self {
        let games = engines.iter().map(|e|
            SingleMcts::new_with_history(e.board.clone(), e.board_history.clone())
        ).collect();
        // Seed from system time — so parallel RustMCTS instances (fsf/lagged
        // create one object per game) don't get identical Dirichlet noise.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdeadbeefcafe1234u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(0xbf58476d1ce4e5b9);
        RustMCTS { games, parallel_sims, rng: seed,
            leaf_game_map: Vec::new(), leaf_counts: Vec::new(), prev_batch: None }
    }

    /// Set the contempt factor on every per-game MCTS. 0.0 = standard play.
    /// Positive values make the search avoid draws; negative values welcome them.
    /// Takes effect on the next selection — accumulated wl/d stay valid.
    pub fn set_contempt(&mut self, contempt: f32) {
        for g in self.games.iter_mut() { g.contempt = contempt; }
    }

    /// Toggle root Dirichlet noise across all per-game trees. False = pure
    /// deterministic policy (eval, FSF, lagged play). Default true (self-play).
    /// Existing root noise is not retracted — change applies to future root
    /// expansions and `make_move` shifts.
    pub fn set_add_dirichlet(&mut self, enabled: bool) {
        for g in self.games.iter_mut() { g.add_dirichlet = enabled; }
    }

    /// Collects leaves for inference.
    /// Returns 2D NumPy array of shape (N, TOTAL_INPUT_PLANES*80) = (N, 11120) with history planes.
    /// target_sims_per_game = total simulations planned for each game
    /// (for honest sims_remaining estimation in best_move_is_decided).
    #[pyo3(signature = (target_sims_per_game=0))]
    pub fn collect_leaves<'py>(&mut self, py: Python<'py>, target_sims_per_game: i32) -> Bound<'py, PyArray2<f32>> {
        self.leaf_game_map.clear();
        let mut new_counts = vec![0usize; self.games.len()];
        let mut flat: Vec<f32> = Vec::new();
        let mut total = 0usize;

        for (g, game) in self.games.iter_mut().enumerate() {
            if game.is_over() { continue; }
            // Accumulating tree reuse: count remaining from sims DONE THIS MOVE,
            // not from full root.visits. Otherwise visits inherited via tree reuse
            // artificially deflate sims_remaining → best_move_is_decided cuts off search
            // prematurely ("choking"). target_sims_per_game=0 = don't use early-stop.
            let visits_so_far = game.arena.get(game.root).visits;
            if target_sims_per_game > 0 {
                let sims_done_this_move = (visits_so_far - game.move_start_visits).max(0);
                let sims_remaining = (target_sims_per_game - sims_done_this_move).max(0);
                if game.best_move_is_decided(sims_remaining) { continue; }
            }
            // Use buffered collection without extra Vec<Vec<f32>> allocations
            let count = game.collect_leaves_into_buf(self.parallel_sims, &mut self.rng);
            new_counts[g] = count;
            for _ in 0..count { self.leaf_game_map.push(g); }
            total += count;
            flat.extend_from_slice(&game.leaf_tensor_buf);
        }

        // Save leaf_counts of THIS batch to prev_batch BEFORE overwriting self.leaf_counts.
        // apply_inference_buffered takes counts from Python (passed after collect_leaves),
        // so policies always go to the correct games in double-buffering.
        self.prev_batch = Some(BatchState { leaf_counts: new_counts.clone() });
        self.leaf_counts = new_counts;

        // Single tensor size = TOTAL_INPUT_PLANES * 80 (139 * 80 = 11120 with history planes).
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

    /// Hashes of current pending leaves (8 bytes per leaf, in same order as collect_leaves).
    /// Used for NN transposition cache: position key = 64-bit hash,
    /// history-independent (LC0 kCacheHistoryLength=0). Previously Python cached by
    /// full tensor bytes (44KB key) — slow + same position with different history
    /// gave different keys → nearly 0% hit rate.
    pub fn get_leaf_hashes(&self) -> Vec<u64> {
        let mut hashes = Vec::new();
        for game in &self.games {
            for board in &game.pending_boards {
                hashes.push(compute_board_hash(board));
            }
        }
        hashes
    }

    /// Applies GPU inference results to trees.
    /// Accepts NumPy arrays directly — zero serialization overhead.
    /// policies: shape (N, 7000) f32
    /// values:   shape (N,)      f32 — Q = P(W) - P(L)
    /// draws:    shape (N,)      f32 — P(D), for D-head tracking in nodes
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

    /// Returns leaf_counts of current batch — Python saves and passes
    /// to apply_inference_buffered for double-buffering.
    pub fn get_current_batch_counts(&self) -> Vec<usize> {
        self.leaf_counts.clone()
    }

    /// Removes accumulated virtual_loss and clears pending for games in range [from, to).
    /// Needed on error paths: otherwise vloss would remain forever and poison the search.
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

    /// apply_inference with explicit batch_counts for correct double-buffering.
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
            // Python/Rust state machine desync. Silent return would leave vloss
            // on pending leaves forever → search tree poisoned for the rest of the game.
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

    /// Final policy vectors from visit counts.
    pub fn get_policies(&self) -> Vec<Vec<f32>> {
        self.games.iter().map(|g| g.get_policy()).collect()
    }

    /// Root value estimates (Q = P(W) - P(L)).
    pub fn get_values(&self) -> Vec<f32> {
        self.games.iter().map(|g| g.root_value()).collect()
    }

    /// Root draw probabilities (D = P(Draw)). Used for WDL-based resign:
    /// P(L) = (1 - Q - D) / 2 — more precise than Q threshold (distinguishes "sure draw" vs "loss").
    pub fn get_draws(&self) -> Vec<f32> {
        self.games.iter().map(|g| g.root_draw()).collect()
    }

    /// Game completion status.
    pub fn games_over(&mut self) -> Vec<bool> {
        self.games.iter_mut().map(|g| g.is_over()).collect()
    }

    /// Applies a move to a specific game (for tree reuse).
    /// After root shift — add fresh Dirichlet noise, otherwise exploration
    /// drops to zero starting from the 2nd half-move of the game.
    pub fn make_move(&mut self, game_idx: usize, m_int: u32) {
        if game_idx < self.games.len() {
            self.games[game_idx].make_move(m_int);
            // Re-noise only when the MCTS was constructed in self-play mode;
            // in eval/FSF/lagged add_dirichlet=false the new root should keep
            // a deterministic policy distribution.
            if self.games[game_idx].add_dirichlet {
                let rng = &mut self.rng;
                self.games[game_idx].renoise_root(rng);
            }
            // root changed → old snapshot is invalid
            self.games[game_idx].kld_reset();
        }
    }

    /// KLD-early-exit gate. Computes max KL(prev || curr) across all games
    /// and IMMEDIATELY saves current as new snapshot.
    ///
    /// Returns f32::INFINITY if some games have no snapshot — first call
    /// after make_move/init always returns +inf (Python should skip the gate
    /// on first check and only use results from the second+ call).
    ///
    /// All compute happens in Rust → marshalling cost = 1 float instead of 7000*N_games.
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

    /// Reset snapshots for all games. Needed manually if external code changed the root.
    pub fn kld_reset_all(&mut self) {
        for g in &mut self.games { g.kld_reset(); }
    }

    pub fn num_games(&self) -> usize { self.games.len() }
    pub fn last_batch_size(&self) -> usize { self.leaf_game_map.len() }
}
