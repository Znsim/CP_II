"""
Dwell(유지) 입력 기반 실시간 단어 조합기
"""
import time
from .korean_composer import KoreanComposer

DWELL_SECS  = 1.0   # 자모 확정에 필요한 유지 시간
SPACE_SECS  = 2.0   # none 유지 시 공백 추가
COOLDOWN    = 0.5   # 확정 직후 동일 자모 재입력 방지


class WordBuilder:
    """
    realtime_spelling_demo.py 에서 매 프레임 update(label) 를 호출하면
    Dwell 타이머로 자모를 확정하고 KoreanComposer 로 단어를 조합한다.
    """

    def __init__(self, dwell: float = DWELL_SECS,
                 space_dwell: float = SPACE_SECS,
                 cooldown: float = COOLDOWN):
        self.dwell       = dwell
        self.space_dwell = space_dwell
        self.cooldown    = cooldown
        self.composer    = KoreanComposer()

        self._cur_label      = None
        self._label_start    = 0.0
        self._last_committed = None
        self._committed_at   = 0.0

    # ── public ────────────────────────────────────────────────────────

    def update(self, label: str | None) -> dict:
        """
        매 프레임 호출.

        Args:
            label: 현재 인식된 자모 문자열, 없으면 None 또는 'none'

        Returns:
            {
                'text':      전체 확정 텍스트 + 조합 중 음절,
                'composing': 조합 중인 미확정 음절,
                'progress':  dwell 진행도 0.0 ~ 1.0,
                'committed': 이번 프레임에 새로 확정된 자모 (없으면 None),
            }
        """
        now = time.time()
        label = label if (label and label != 'none') else None

        if label != self._cur_label:
            self._cur_label      = label
            self._label_start    = now
            self._last_committed = None   # 제스처 바뀌면 동일 자모 재입력 허용

        elapsed     = now - self._label_start
        can_commit  = (now - self._committed_at) > self.cooldown
        committed   = None

        if label is None:
            progress = min(elapsed / self.space_dwell, 1.0)
            if elapsed >= self.space_dwell and can_commit and self.composer.composing:
                self.composer.space()
                self._committed_at   = now
                self._last_committed = None
                committed = ' '
        else:
            progress = min(elapsed / self.dwell, 1.0)
            if elapsed >= self.dwell and can_commit and label != self._last_committed:
                self.composer.add(label)
                self._last_committed = label
                self._committed_at   = now
                committed = label

        return {
            'text':      self.composer.text,
            'composing': self.composer.composing,
            'progress':  progress,
            'committed': committed,
        }

    def backspace(self):
        self.composer.backspace()
        self._last_committed = None

    def clear(self):
        self.composer.clear()
        self._last_committed = None
        self._cur_label      = None
