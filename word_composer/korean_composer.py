"""
한글 자모 → 음절 조합 엔진 (표준 두벌식 오토마타)
"""

CHOSUNG  = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSUNG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

CONSONANTS = set(CHOSUNG)
VOWELS     = set(JUNGSUNG)

# 두 단순 모음 → 복합 모음
COMPOUND_VOWEL = {
    ('ㅗ', 'ㅏ'): 'ㅘ',
    ('ㅗ', 'ㅐ'): 'ㅙ',
    ('ㅗ', 'ㅣ'): 'ㅚ',
    ('ㅜ', 'ㅓ'): 'ㅝ',
    ('ㅜ', 'ㅔ'): 'ㅞ',
    ('ㅜ', 'ㅣ'): 'ㅟ',
    ('ㅡ', 'ㅣ'): 'ㅢ',
}

# 두 단순 자음 → 복합 종성
COMPOUND_JONG = {
    ('ㄱ', 'ㅅ'): 'ㄳ',
    ('ㄴ', 'ㅈ'): 'ㄵ',
    ('ㄴ', 'ㅎ'): 'ㄶ',
    ('ㄹ', 'ㄱ'): 'ㄺ',
    ('ㄹ', 'ㅁ'): 'ㄻ',
    ('ㄹ', 'ㅂ'): 'ㄼ',
    ('ㄹ', 'ㅅ'): 'ㄽ',
    ('ㄹ', 'ㅌ'): 'ㄾ',
    ('ㄹ', 'ㅍ'): 'ㄿ',
    ('ㄹ', 'ㅎ'): 'ㅀ',
    ('ㅂ', 'ㅅ'): 'ㅄ',
}

# 같은 자음 두 번 → 쌍자음
DOUBLE_CONSONANT = {
    'ㄱ': 'ㄲ',
    'ㄷ': 'ㄸ',
    'ㅂ': 'ㅃ',
    'ㅅ': 'ㅆ',
    'ㅈ': 'ㅉ',
}
SPLIT_JONG = {v: k for k, v in COMPOUND_JONG.items()}

# 복합 모음 → 첫 번째 단순 모음 (backspace용)
VOWEL_FIRST = {v: k[0] for k, v in COMPOUND_VOWEL.items()}


def _compose(cho: str, jung: str, jong: str = '') -> str:
    return chr(0xAC00 + (CHOSUNG.index(cho) * 21 + JUNGSUNG.index(jung)) * 28 + JONGSUNG.index(jong))


def _decompose(syllable: str):
    """완성형 음절 → (초성, 중성, 종성) 튜플"""
    code = ord(syllable) - 0xAC00
    jong_i = code % 28
    jung_i = (code // 28) % 21
    cho_i  = code // 28 // 21
    return CHOSUNG[cho_i], JUNGSUNG[jung_i], JONGSUNG[jong_i]


class KoreanComposer:
    """
    지문자 자모를 두벌식 오토마타로 조합해 완성형 한글을 생성한다.

    사용법:
        c = KoreanComposer()
        for jamo in ['ㅅ', 'ㅏ', 'ㄱ', 'ㅗ', 'ㅏ']:
            c.add(jamo)
        print(c.text)   # → '사과'
    """

    def __init__(self):
        self._done: list[str] = []
        self._cho:  str | None = None
        self._jung: str | None = None
        self._jong: str | None = None   # 단순 or 복합 종성

    # ── public ────────────────────────────────────────────────────────

    def add(self, jamo: str):
        """자모 하나 추가"""
        if jamo in CONSONANTS:
            self._input_consonant(jamo)
        elif jamo in VOWELS:
            self._input_vowel(jamo)

    def space(self):
        """현재 조합 중인 음절을 확정하고 공백 추가"""
        self._commit()
        self._done.append(' ')

    def backspace(self):
        """마지막으로 입력된 자모 하나 취소"""
        if self._jong is not None:
            if self._jong in SPLIT_JONG:            # 복합 종성 → 첫째 자음만 남김
                self._jong = SPLIT_JONG[self._jong][0]
            else:
                self._jong = None
        elif self._jung is not None:
            if self._jung in VOWEL_FIRST:           # 복합 모음 → 첫째 모음으로
                self._jung = VOWEL_FIRST[self._jung]
            else:
                self._jung = None
        elif self._cho is not None:
            self._cho = None
        elif self._done:
            last = self._done.pop()
            if last == ' ':
                return
            code = ord(last) - 0xAC00
            if 0 <= code < 11172:                   # 완성형 음절 분해
                cho, jung, jong = _decompose(last)
                self._cho  = cho
                self._jung = jung
                self._jong = jong if jong else None

    def clear(self):
        """전체 초기화"""
        self._done.clear()
        self._cho = self._jung = self._jong = None

    @property
    def text(self) -> str:
        """확정된 텍스트 + 현재 조합 중인 음절"""
        return ''.join(self._done) + self._current()

    @property
    def composing(self) -> str:
        """현재 조합 중인 미확정 음절"""
        return self._current()

    # ── internal ──────────────────────────────────────────────────────

    def _current(self) -> str:
        if self._cho is None:
            return ''
        if self._jung is None:
            return self._cho
        return _compose(self._cho, self._jung, self._jong or '')

    def _commit(self):
        cur = self._current()
        if cur:
            self._done.append(cur)
        self._cho = self._jung = self._jong = None

    def _input_consonant(self, c: str):
        if self._jung is None:
            # 모음 없음: 쌍자음 시도 후 초성 처리
            if self._cho is not None and self._cho == c and c in DOUBLE_CONSONANT:
                self._cho = DOUBLE_CONSONANT[c]
            elif self._cho is not None:
                self._done.append(self._cho)
                self._cho = c
            else:
                self._cho = c
            self._jong = None
        else:
            # 모음 있음: 종성 자리
            if self._jong is None:
                self._jong = c
            elif self._jong == c and c in DOUBLE_CONSONANT:
                self._jong = DOUBLE_CONSONANT[c]            # 종성 쌍자음 (ㄲ, ㅆ)
            else:
                comp = COMPOUND_JONG.get((self._jong, c))
                if comp:
                    self._jong = comp
                else:
                    self._done.append(_compose(self._cho, self._jung, self._jong))
                    self._cho  = c
                    self._jung = self._jong = None

    def _input_vowel(self, v: str):
        if self._cho is None:
            # 빈 상태: ㅇ(묵음) + 모음
            self._cho  = 'ㅇ'
            self._jung = v
        elif self._jung is None:
            # 초성만 있음: 중성 추가
            self._jung = v
        elif self._jong is None:
            # 초성+중성, 종성 없음: 복합 모음 시도
            comp = COMPOUND_VOWEL.get((self._jung, v))
            if comp:
                self._jung = comp
            else:
                # 현재 음절 확정, 새 ㅇ+모음
                self._done.append(_compose(self._cho, self._jung))
                self._cho  = 'ㅇ'
                self._jung = v
                self._jong = None
        else:
            # 종성 있음 → 종성이 다음 초성으로 이동 (핵심 규칙)
            if self._jong in SPLIT_JONG:
                # 복합 종성: 첫째는 종성으로 확정, 둘째가 다음 초성
                first, second = SPLIT_JONG[self._jong]
                self._done.append(_compose(self._cho, self._jung, first))
                self._cho = second
            else:
                # 단순 종성: 종성이 다음 초성으로
                self._done.append(_compose(self._cho, self._jung))
                self._cho = self._jong
            self._jung = v
            self._jong = None


# ── 간단 테스트 ────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        (['ㅅ', 'ㅏ', 'ㄱ', 'ㅗ', 'ㅏ'],          '사과'),
        (['ㅎ', 'ㅏ', 'ㄴ', 'ㄱ', 'ㅡ', 'ㄹ'],      '한글'),
        (['ㅅ', 'ㅜ', 'ㅇ', 'ㅓ'],                 '수어'),
        (['ㄴ', 'ㅏ', 'ㄹ', 'ㄱ', 'ㅐ'],           '날개'),
        (['ㄷ', 'ㅏ', 'ㄹ', 'ㄱ'],                '닭'),
        # 쌍자음 테스트
        (['ㄱ', 'ㄱ', 'ㅜ', 'ㅁ'],                '꿈'),
        (['ㅅ', 'ㅅ', 'ㅏ', 'ㅇ', 'ㅡ', 'ㅣ'],      '싸의'),  # ㅇ받침 + 모음 → 받침이 다음 초성으로 분리
        (['ㅅ', 'ㅅ', 'ㅏ', 'ㄷ', 'ㅏ'],           '싸다'),  # 연속 입력으로 쌍자음
        (['ㅂ', 'ㅂ', 'ㅏ', 'ㄹ', 'ㄹ', 'ㅣ'],      '빨리'),  # 초성+종성 쌍자음
        (['ㅎ', 'ㅏ', 'ㄴ', 'ㄱ', 'ㅡ', 'ㄹ'],      '한글'),  # 받침 분리 정상 동작
    ]

    for jamos, expected in tests:
        c = KoreanComposer()
        for j in jamos:
            c.add(j)
        c._commit()
        result = c.text
        status = '✅' if result == expected else '❌'
        print(f'{status}  {"+".join(jamos):25s} → {result:6s}  (기대: {expected})')
