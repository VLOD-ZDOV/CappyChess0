pub struct Bitboard(pub u128);
impl Bitboard {
    const BOARD_MASK: u128 = (1u128 << 80) - 1;

    pub fn new(val: u128) -> Self {
        Bitboard(val & Self::BOARD_MASK)
    }

    // Корректные ходы коня для доски 10x8 с проверкой краев
    pub fn knight_moves(sq: u8) -> u128 {
        let b = 1u128 << sq;
        let mut m = 0u128;
        let file = sq % 10;

        // Сдвиги: ±8, ±10, ±11, ±12 (адаптировано под ширину 10)
        if file >= 2 { m |= (b >> 12) | (b << 8); }
        if file >= 1 { m |= (b >> 21) | (b << 19); } // -21 = -2*10-1
        if file <= 8 { m |= (b >> 19) | (b << 21); }
        if file <= 7 { m |= (b >> 8) | (b << 12); }

        m & Self::BOARD_MASK
    }

    // Заготовки для слонов/ладей (в продакшене используйте Magic Bitboards или PEXT)
    pub fn bishop_attacks(_sq: u8, _occ: u128) -> u128 { 0 }
    pub fn rook_attacks(_sq: u8, _occ: u128) -> u128 { 0 }

    pub fn archbishop_moves(sq: u8, occ: u128) -> u128 {
        Self::bishop_attacks(sq, occ) | Self::knight_moves(sq)
    }

    pub fn chancellor_moves(sq: u8, occ: u128) -> u128 {
        Self::rook_attacks(sq, occ) | Self::knight_moves(sq)
    }
}
