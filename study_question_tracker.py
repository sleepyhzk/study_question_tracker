"""
Study Question Tracker - Screenshot -> Gemini -> Answer + Explanation
====================================================================
Sử dụng gói SDK mới: google-genai

v3.5: Quản lý/xóa ảnh chụp, xóa ảnh theo khung giờ, chụp bằng nút chuột (mặc định chuột phải).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import queue
import re
import threading
import logging
import warnings
import time
import uuid
import tkinter as tk
from pathlib import Path
from typing import Any, Callable

import io
import sys
import socket

# Windows: bật DPI-awareness để toạ độ chọn vùng khớp với toạ độ chụp màn hình
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import customtkinter as ctk
import mss
from PIL import Image, ImageDraw, ImageEnhance, ImageTk

try:
    import pystray
except Exception:  # pragma: no cover
    pystray = None
from pynput import keyboard, mouse

# --- Thư viện cho chế độ "Nghe" (tùy chọn) --------------------------------- #
import wave

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - thiếu lib hoặc không có thiết bị âm thanh
    sd = None

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

# Chép lời OFFLINE trên máy (không tốn lượt gọi API): pip install faster-whisper
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover
    WhisperModel = None

# Fallback chép lời qua speech_recognition (Google Web Speech - cần mạng): pip install SpeechRecognition
try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None

# Giới hạn kích thước gói tin gửi lên Gemini (tránh 429 do quá nhiều token)
MAX_LISTEN_CHARS = 1200          # transcript gửi kèm: tối đa ~1200 ký tự (~300 token)
LISTEN_RECENT_SEC = 120.0        # chỉ lấy lời nghe trong 2 phút gần nhất
LISTEN_KEYWORD_SEC = 300.0       # câu chứa từ khoá câu hỏi: lấy thêm trong 5 phút
QUESTION_KEYWORDS = (
    "?", "what", "which", "who", "where", "when", "why", "how", "true", "false",
    "question", "answer", "choose", "select", "correct",
    "câu hỏi", "câu", "đáp án", "chọn", "đúng", "sai", "tại sao", "vì sao", "bao nhiêu",
    "ở đâu", "khi nào", "ai ", "cái gì", "là gì", "như thế nào",
)
MAX_AUDIO_SECONDS = 60.0         # nếu phải gửi audio thô: tối đa 60 giây gần nhất
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MAX_IMAGE_SIDE = 1600            # thu nhỏ ảnh quét, cạnh dài tối đa 1600px


def _wav_bytes_to_float32(wav_bytes: bytes):
    """Giải mã WAV bằng thư viện chuẩn (wave) -> mảng float32.

    Tránh hoàn toàn lỗi 'open() got an unexpected keyword argument metadata_errors'
    do xung đột phiên bản faster-whisper / PyAV khi tự giải mã file.
    """
    if np is None:
        raise RuntimeError("Thiếu numpy. Chạy: pip install numpy")
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Định dạng WAV {width * 8}-bit chưa hỗ trợ.")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000 and rate > 0:
        # Resample tuyến tính đơn giản về 16kHz (đủ tốt cho whisper)
        n_new = int(len(data) * 16000.0 / rate)
        if n_new > 0:
            data = np.interp(np.linspace(0, len(data) - 1, n_new),
                             np.arange(len(data)), data).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


class LocalTranscriber:
    """Speech-to-text chạy ngay trên máy bằng faster-whisper. Không gọi API.

    Nếu faster-whisper lỗi/không cài được, tự chuyển sang speech_recognition
    (Google Web Speech - cần mạng) thay vì crash.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self.error = ""
        self._use_sr_fallback = False

    @property
    def available(self) -> bool:
        return (WhisperModel is not None and not self.error) or sr is not None

    @property
    def engine_name(self) -> str:
        if self._use_sr_fallback:
            return "speech_recognition (online)"
        return "faster-whisper (offline)"

    def transcribe(self, wav_bytes: bytes, language: str = "auto") -> str:
        with self._lock:
            if not self._use_sr_fallback and WhisperModel is not None and not self.error:
                try:
                    if self._model is None:
                        self._model = WhisperModel("small", device="cpu", compute_type="int8")
                    lang = None if language == "auto" else language
                    audio = _wav_bytes_to_float32(wav_bytes)
                    segments, _info = self._model.transcribe(audio, language=lang,
                                                             vad_filter=True, beam_size=1)
                    return " ".join(seg.text.strip() for seg in segments).strip()
                except Exception as exc:
                    # Không crash: ghi nhận lỗi và chuyển sang fallback nếu có
                    self.error = str(exc)
                    print("Chép lời offline lỗi (chuyển sang fallback):", exc)
                    if sr is not None:
                        self._use_sr_fallback = True
                    else:
                        return ""
            return self._transcribe_sr(wav_bytes, language)

    def _transcribe_sr(self, wav_bytes: bytes, language: str) -> str:
        """Fallback: speech_recognition + Google Web Speech (cần internet)."""
        if sr is None:
            return ""
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                audio = recognizer.record(source)
            lang_code = {"vi": "vi-VN", "en": "en-US"}.get(language, "vi-VN")
            return (recognizer.recognize_google(audio, language=lang_code) or "").strip()
        except sr.UnknownValueError:
            return ""
        except Exception as exc:
            print("Chép lời fallback lỗi:", exc)
            return ""


# --------------------------------------------------------------------------- #
# Đường dẫn & cấu hình
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
HISTORY_PATH = BASE_DIR / "history.json"
CHAT_HISTORY_PATH = BASE_DIR / "chat_history.json"
SHOTS_DIR = BASE_DIR / "screenshots"
SHOTS_DIR.mkdir(exist_ok=True)

MODEL_CHOICES = [
    "gemini-3.8-flash",        # mặc định mới (Google đã ngừng cấp gemini-2.5-* cho tài khoản mới)
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",   # nhẹ, rẻ nhất
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.1-pro-preview",  # mạnh nhất nhưng KHÔNG có gói miễn phí
    "gemini-3-pro-preview",
]
# Thứ tự dự phòng mặc định: mỗi model có hạn mức miễn phí riêng, hết model này thì dùng model kế tiếp.
OLD_DEFAULT_FALLBACKS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                         "gemini-3-flash-preview"]
DEFAULT_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash",
                     "gemini-3-flash-preview"]
QUOTA_PATH = BASE_DIR / "quota.json"

DEFAULT_MODEL = MODEL_CHOICES[0]


def normalize_model(name: Any) -> str:
    """Chỉ chấp nhận model đời 3; mọi giá trị cũ trong config.json -> mặc định."""
    name = (str(name) if name else "").strip()
    return name if name in MODEL_CHOICES else DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Kho mật khẩu: API key được lưu trong Windows Credential Manager, KHÔNG nằm trong config.json
# --------------------------------------------------------------------------- #
KEYSTORE_TARGET = "StudyQuestionTracker/gemini_api_key"
_WIN_CRED: dict[str, Any] = {}


def _win_cred_api() -> dict[str, Any]:
    if _WIN_CRED:
        return _WIN_CRED
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", FILETIME), ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR),
        ]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
    advapi.CredWriteW.restype = wintypes.BOOL
    advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    advapi.CredDeleteW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]
    advapi.CredFree.restype = None
    _WIN_CRED.update(ctypes=ctypes, CREDENTIALW=CREDENTIALW, advapi=advapi)
    return _WIN_CRED


def secret_available() -> bool:
    if sys.platform == "win32":
        try:
            _win_cred_api()
            return True
        except Exception:
            return False
    try:
        import keyring  # noqa: F401
        return True
    except Exception:
        return False


def secret_get() -> str:
    try:
        if sys.platform == "win32":
            api = _win_cred_api()
            ct, advapi = api["ctypes"], api["advapi"]
            pcred = ct.POINTER(api["CREDENTIALW"])()
            if not advapi.CredReadW(KEYSTORE_TARGET, 1, 0, ct.byref(pcred)):
                return ""
            try:
                size = pcred.contents.CredentialBlobSize
                raw = ct.string_at(pcred.contents.CredentialBlob, size)
                return raw.decode("utf-16-le")
            finally:
                advapi.CredFree(pcred)
        import keyring
        return keyring.get_password("StudyQuestionTracker", "gemini_api_key") or ""
    except Exception:
        return ""


def secret_set(value: str) -> bool:
    """Lưu API key vào kho mật khẩu và đọc lại để chắc chắn đã lưu được. True nếu thành công."""
    if not _secret_write(value):
        return False
    return secret_get() == value


def _secret_write(value: str) -> bool:
    try:
        if sys.platform == "win32":
            api = _win_cred_api()
            ct, advapi = api["ctypes"], api["advapi"]
            if not value:
                advapi.CredDeleteW(KEYSTORE_TARGET, 1, 0)   # chưa có thì cũng coi như xong
                return True
            blob = value.encode("utf-16-le")
            buf = (ct.c_ubyte * len(blob)).from_buffer_copy(blob)
            cred = api["CREDENTIALW"]()
            cred.Type = 1                      # CRED_TYPE_GENERIC
            cred.TargetName = KEYSTORE_TARGET
            cred.UserName = "gemini_api_key"
            cred.CredentialBlobSize = len(blob)
            cred.CredentialBlob = ct.cast(buf, ct.POINTER(ct.c_ubyte))
            cred.Persist = 2                   # CRED_PERSIST_LOCAL_MACHINE
            return bool(advapi.CredWriteW(ct.byref(cred), 0))
        import keyring
        if value:
            keyring.set_password("StudyQuestionTracker", "gemini_api_key", value)
        else:
            try:
                keyring.delete_password("StudyQuestionTracker", "gemini_api_key")
            except Exception:
                pass
        return True
    except Exception:
        return False


GITIGNORE_LINES = [
    "# Study Question Tracker - dữ liệu cá nhân, KHÔNG đưa lên GitHub",
    "config.json", "history.json", "chat_history.json", "quota.json", "*.tmp",
    "screenshots/", "audio/", "app_icon.ico", "__pycache__/", "*.pyc",
]


def ensure_gitignore() -> None:
    """Tạo/bổ sung .gitignore để config.json (và dữ liệu cá nhân) không bị đẩy lên GitHub nhầm."""
    try:
        path = BASE_DIR / ".gitignore"
        existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        missing = [line for line in GITIGNORE_LINES if line not in existing]
        if missing:
            with path.open("a", encoding="utf-8") as fh:
                if existing and existing[-1].strip():
                    fh.write("\n")
                fh.write("\n".join(missing) + "\n")
    except Exception:
        pass


DEFAULT_CONFIG: dict[str, Any] = {
    "sidebar_hidden": False,     # ẩn/hiện cột Lịch sử quét bên trái
    "chat_list_hidden": False,   # ẩn/hiện danh sách chat AI
    "api_key": "",
    "model": DEFAULT_MODEL,
    "rpm_limit": 5,               # lượt/phút tối đa cho mỗi model (gói miễn phí = 5; 0 = không giới hạn)
    "auto_fallback": True,        # hết lượt model chính thì tự chuyển sang model dự phòng
    "model_fallbacks": list(DEFAULT_FALLBACKS),
    "ai_speed": "fast",          # fast = trả lời nhanh (suy nghĩ ít) | accurate = suy nghĩ kỹ, chậm hơn
    "hotkey": "<ctrl>+<shift>+x",
    "service_hotkey": "<ctrl>+<shift>+k",  # phím tắt bật/tắt dịch vụ
    "listen_hotkey": "<ctrl>+<shift>+l",  # phím tắt bật/tắt chế độ nghe
    "minimize_to_tray": True,  # thu nhỏ (–) -> chạy ngầm ở khay; nút X luôn thoát hẳn
    "region": None,
    "scan_mode": "select",  # select | fixed | full
    "save_screenshots": True,
    "mouse_trigger": True,
    "mouse_button": "right",  # right | middle | x1 | x2
    "mouse_clicks": 2,        # 1 = nhấn 1 lần, 2 = nhấp đúp (2 lần liên tiếp) thì tự quét
    # --- Chế độ nghe audio --- #
    "listen_device": "",          # tên thiết bị input (mic hoặc Stereo Mix/loopback)
    "listen_language": "auto",    # auto | vi | en
    "listen_threshold": 0.012,    # ngưỡng âm lượng coi là có tiếng nói (RMS)
    "listen_silence": 1.2,        # số giây im lặng để chốt một câu hỏi
    "listen_auto_stop": 15,       # tự tắt ghi âm nếu im lặng liên tục (giây); 0 = không tự tắt
    "listen_min_speech": 1.0,     # độ dài tối thiểu của đoạn nói (giây)
    "listen_save_audio": False,
    "listen_context_sec": 180,
}

AUDIO_DIR = BASE_DIR / "audio"
AUDIO_SAMPLERATE = 16000

LANG_HINT = {
    "auto": "Tự nhận diện ngôn ngữ trong audio.",
    "vi": "Audio chủ yếu là tiếng Việt.",
    "en": "Audio chủ yếu là tiếng Anh.",
}

PROMPT_AUDIO = (
    "Bạn là trợ giảng. Đoạn audio dưới đây là một câu hỏi (có thể là bài nghe trắc nghiệm).\n"
    "{lang_hint}\n"
    "Bước 1: nghe và chép lại (transcribe) nguyên văn nội dung audio.\n"
    "Bước 2: trả lời câu hỏi đó.\n"
    "Trả về đúng định dạng markdown sau:\n\n"
    "**Transcript:** <nguyên văn nội dung nghe được>\n\n"
    "**Câu hỏi:** <câu hỏi được hỏi>\n\n"
    "**Lựa chọn:**\n"
    "A. <nội dung A>\nB. <nội dung B>\nC. <nội dung C>\nD. <nội dung D>\n\n"
    "**Đáp án đúng:** <chỉ ghi chữ cái A/B/C/D>\n\n"
    "**Giải thích:** <giải thích ngắn gọn, dễ hiểu, bằng tiếng Việt>\n\n"
    "Nếu audio không có đủ 4 lựa chọn, hãy tự đề xuất 4 phương án hợp lý và đánh dấu đáp án đúng.\n"
    "Nếu audio chỉ là tiếng ồn hoặc không có câu hỏi, hãy trả lời đúng một dòng: KHÔNG CÓ CÂU HỎI"
)

PROMPT_TRANSCRIBE = (
    "Hãy chép lại (speech-to-text) nguyên văn lời nói trong đoạn audio. {lang_hint}\n"
    "Chỉ trả về đúng văn bản đã nghe được, không thêm lời bình, không định dạng markdown.\n"
    "Nếu không có lời nói rõ ràng (chỉ tiếng ồn/nhạc/im lặng), trả về đúng: [KHÔNG CÓ LỜI NÓI]"
)

PROMPT_LISTEN_CONTEXT = (
    "\n\nNGỮ CẢNH BỔ SUNG — nội dung bài nghe (đã chép từ audio) ngay trước khi chụp màn hình:\n"
    "\"\"\"\n{transcript}\n\"\"\"\n"
    "Kết hợp cả ảnh câu hỏi và nội dung bài nghe để chọn đáp án chính xác. "
    "Nêu ngắn gọn chi tiết bài nghe dẫn tới đáp án trong phần Giải thích."
)

PROMPT_LISTEN_AUDIO = (
    "\n\nKèm theo đây là các đoạn AUDIO bài nghe đã được ghi lại ngay trước khi chụp màn hình "
    "(theo đúng thứ tự thời gian). Hãy nghe toàn bộ các đoạn audio này, chép lại nội dung và "
    "kết hợp với ảnh câu hỏi để chọn đáp án. Thêm mục **Transcript:** đặt trước **Câu hỏi:**."
)

PROMPT_TEXT = (
    "Bạn là trợ giảng. Dưới đây là nội dung một câu hỏi dạng văn bản.\n"
    "Hãy trả lời theo đúng định dạng markdown sau:\n\n"
    "**Câu hỏi:** <chép lại câu hỏi>\n\n"
    "**Lựa chọn:**\nA. <...>\nB. <...>\nC. <...>\nD. <...>\n\n"
    "**Đáp án đúng:** <chỉ ghi chữ cái A/B/C/D>\n\n"
    "**Giải thích:** <giải thích chi tiết, ngắn gọn, bằng tiếng Việt>\n\n"
    "Nội dung câu hỏi:\n{content}"
)

PROMPT_SOLVE = (
    "Bạn là trợ giảng. Ảnh dưới đây chụp một câu hỏi trắc nghiệm.\n"
    "Hãy trả lời bằng tiếng Việt theo đúng định dạng markdown cấu trúc sau để hệ thống bóc tách:\n\n"
    "**Câu hỏi:** <chép lại nguyên văn câu hỏi trong ảnh>\n\n"
    "**Lựa chọn:**\n"
    "A. <nội dung lựa chọn A>\n"
    "B. <nội dung lựa chọn B>\n"
    "C. <nội dung lựa chọn C>\n"
    "D. <nội dung lựa chọn D>\n\n"
    "**Đáp án đúng:** <Chỉ ghi rõ chữ cái đáp án đúng, ví dụ: A hoặc B hoặc C hoặc D>\n\n"
    "**Giải thích:** <giải thích chi tiết từng bước, ngắn gọn, dễ hiểu>\n\n"
    "Nếu ảnh không chứa câu hỏi trắc nghiệm, hãy tự động trích xuất nội dung và định dạng tương ứng nhất."
)


# Ẩn các log cảnh báo thông tin không cần thiết của SDK trên terminal
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mss")


def _new_mss():
    """mss >= 10 dùng mss.MSS(); bản cũ chỉ có mss.mss()."""
    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


def safe_after(widget: Any, ms: int, fn: Callable[..., Any], *args: Any) -> str | None:
    """Thay thế an toàn cho widget.after (Tk).

    Lỗi "Misc.after.<locals>.callit() takes 0 positional arguments" xảy ra khi Tcl
    gọi lại hàm nội bộ callit() kèm tham số thừa (thường do after() bị gọi từ luồng
    phụ như pystray/keyboard, hoặc qua customtkinter). Ở đây ta tự đăng ký một lệnh
    Tcl nhận *args tuỳ ý nên không bao giờ lỗi số tham số.
    """
    try:
        if not widget.winfo_exists():
            return None
    except Exception:
        return None
    name_box: list[str] = []

    def _cb(*_ignored: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:  # không để lỗi callback làm sập vòng lặp Tk
            print("after() callback error:", exc)
        finally:
            try:
                widget.deletecommand(name_box[0])
            except Exception:
                pass

    name = widget.register(_cb)
    name_box.append(name)
    return widget.tk.call("after", max(0, int(ms)), name)


def safe_after_cancel(widget: Any, job: str | None) -> None:
    if not job:
        return
    try:
        widget.tk.call("after", "cancel", job)
    except Exception:
        pass


def load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return fallback


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Gemini client (sử dụng google.genai mới nhất)
# --------------------------------------------------------------------------- #
def _gen_config(**extra: Any):
    """Cấu hình chuẩn: tắt automatic function calling (không dùng tools) để hết cảnh báo AFC."""
    if types is None:
        return None
    return types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **extra,
    )


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _compact_image_bytes(image: Image.Image) -> bytes:
    """Thu nhỏ + nén JPEG để gói tin gửi lên nhỏ gọn."""
    img = image.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class ApiBusyError(RuntimeError):
    """Lỗi tạm thời của API (429 quá hạn mức / 503 quá tải) - hiển thị thông báo thân thiện."""


class NetworkError(RuntimeError):
    """Mất kết nối mạng / không phân giải được máy chủ - thông báo thân thiện."""


_NET_OK_UNTIL = 0.0


def check_internet(host: str = "generativelanguage.googleapis.com", port: int = 443,
                   timeout: float = 3.0) -> bool:
    """Kiểm tra kết nối tới máy chủ Gemini; kết quả "có mạng" được nhớ 30 giây để khỏi tốn thêm 1 lượt kết nối."""
    global _NET_OK_UNTIL
    if time.time() < _NET_OK_UNTIL:
        return True
    try:
        with socket.create_connection((host, port), timeout=timeout):
            _NET_OK_UNTIL = time.time() + 30.0
            return True
    except OSError:
        return False


def _is_network_error(exc: Exception) -> bool:
    if isinstance(exc, (socket.gaierror, socket.timeout, ConnectionError, TimeoutError)):
        return True
    t = _err_text(exc)
    return any(k in t for k in ("getaddrinfo", "connecterror", "connection refused",
                                "network is unreachable", "name or service not known",
                                "temporary failure in name resolution", "timed out",
                                "errno 11001", "errno 11002"))


def _err_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def _is_rate_limited(exc: Exception) -> bool:
    t = _err_text(exc)
    return any(k in t for k in ("429", "resource_exhausted", "rate limit", "ratelimit", "quota"))


def _is_overloaded(exc: Exception) -> bool:
    t = _err_text(exc)
    return any(k in t for k in ("503", "unavailable", "overloaded", "500 internal", "deadline"))


def _is_model_unavailable(exc: Exception) -> bool:
    """404: model không còn tồn tại / không còn cấp cho tài khoản mới -> bỏ model này, dùng model khác."""
    t = _err_text(exc)
    return ("404" in t and "not_found" in t) or "no longer available" in t or "is not found" in t


def _is_daily_quota(exc: Exception) -> bool:
    """429 do hết hạn mức THEO NGÀY (chờ vài giây vô ích; phải đổi model hoặc đợi sang ngày mới)."""
    t = _err_text(exc).replace(" ", "").replace("_", "")
    return "perday" in t or "dailylimit" in t or "requestsperday" in t


def _next_quota_reset(after: float) -> float:
    """Hạn mức theo ngày của Gemini reset lúc 0h giờ Thái Bình Dương (~07:00-08:00 UTC = 14h-15h giờ Việt Nam).
    Dùng mốc 07:00 UTC (sớm nhất) để lỡ sớm thì chỉ tốn 1 lần thử lại."""
    dt = _dt.datetime.fromtimestamp(after, _dt.timezone.utc)
    boundary = dt.replace(hour=7, minute=0, second=0, microsecond=0)
    if dt >= boundary:
        boundary += _dt.timedelta(days=1)
    return boundary.timestamp()


class QuotaTracker:
    """Đếm số lượt gọi AI theo từng model/ngày (ước lượng trên máy này) và nhớ model đã hết lượt."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        data = load_json(path, {})
        self.usage: dict[str, dict[str, int]] = data.get("usage", {}) if isinstance(data, dict) else {}
        self.exhausted: dict[str, float] = data.get("exhausted", {}) if isinstance(data, dict) else {}
        self._prune()

    @staticmethod
    def _day_key(ts: float | None = None) -> str:
        # Ngày tính theo giờ Thái Bình Dương (UTC-8 là đủ gần) để khớp với lúc reset hạn mức.
        now = _dt.datetime.fromtimestamp(ts or time.time(), _dt.timezone.utc) - _dt.timedelta(hours=8)
        return now.date().isoformat()

    def _prune(self) -> None:
        today = self._day_key()
        self.usage = {d: v for d, v in self.usage.items() if d >= today}
        now = time.time()
        self.exhausted = {m: t for m, t in self.exhausted.items() if t > now}

    def _save(self) -> None:
        try:
            save_json(self.path, {"usage": self.usage, "exhausted": self.exhausted})
        except Exception:
            pass

    def record(self, model: str) -> None:
        with self._lock:
            day = self.usage.setdefault(self._day_key(), {})
            day[model] = int(day.get(model, 0)) + 1
            self._save()

    def used_today(self, model: str) -> int:
        with self._lock:
            return int(self.usage.get(self._day_key(), {}).get(model, 0))

    def mark_exhausted(self, model: str) -> None:
        with self._lock:
            self.exhausted[model] = _next_quota_reset(time.time())
            self._save()

    def is_exhausted(self, model: str) -> bool:
        with self._lock:
            until = self.exhausted.get(model)
            if until is None:
                return False
            if until <= time.time():
                self.exhausted.pop(model, None)
                self._save()
                return False
            return True

    def summary(self) -> str:
        with self._lock:
            self._prune()
            today = self.usage.get(self._day_key(), {})
            parts = []
            for model in MODEL_CHOICES:
                n = int(today.get(model, 0))
                dead = model in self.exhausted
                if n or dead:
                    parts.append(f"{model}: {n} lượt" + ("  ⛔ hết lượt" if dead else ""))
            return "\n".join(parts) if parts else "Chưa dùng lượt nào hôm nay (tính trên máy này)."


class QuotaExhaustedError(ApiBusyError):
    """Mọi model trong danh sách đều đã hết lượt miễn phí hôm nay."""


def quota_exhausted_message(models: list[str]) -> str:
    return ("Đã hết lượt miễn phí hôm nay của: " + ", ".join(models) + ".\n"
            "Hạn mức được cấp lại lúc 0h giờ Thái Bình Dương (khoảng 14h–15h giờ Việt Nam). "
            "Bạn có thể thêm model dự phòng trong Cài đặt, hoặc bật thanh toán trong Google AI Studio "
            "để có hạn mức cao hơn.")


def _retry_delay_from(exc: Exception) -> float:
    """Đọc retryDelay (giây) do Gemini gợi ý trong thông báo lỗi, nếu có."""
    m = re.search(r"retry[_ ]?delay['\"]?[:= ]+['\"]?(\d+(?:\.\d+)?)", str(exc), re.I)
    if m:
        try:
            return min(60.0, float(m.group(1)))
        except Exception:
            return 0.0
    return 0.0


def friendly_api_error(exc: Exception) -> str:
    if _is_network_error(exc):
        return "Lỗi kết nối mạng, vui lòng kiểm tra lại internet rồi thử lại."
    if _is_rate_limited(exc):
        wait = _retry_delay_from(exc)
        extra = f" (khoảng {int(wait)} giây)" if wait else " vài giây"
        return ("Đang vượt quá giới hạn API (429 - hết hạn mức tạm thời). "
                f"Vui lòng đợi{extra} rồi quét lại.")
    if _is_overloaded(exc):
        return "Máy chủ Gemini đang quá tải. Vui lòng thử lại sau vài giây."
    if _is_model_unavailable(exc):
        return ("Model Gemini đã chọn không còn khả dụng (404). Vào tab Cài đặt, "
                "chọn model khác (vd. gemini-3.8-flash) rồi lưu lại.")
    return str(exc)


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = ""
        self.speed_mode = "fast"
        self.model_name = DEFAULT_MODEL
        self.fallbacks: list[str] = []          # model dự phòng, theo thứ tự ưu tiên
        self.quota: QuotaTracker | None = None
        self.on_switch: Callable[[str, str, bool], None] | None = None   # (model_cũ, model_mới, hết_lượt_ngày)
        self.last_model = ""
        self.rpm_limit = 5                       # 0 = tắt giữ nhịp
        self._calls: dict[str, list[float]] = {}  # thời điểm các lượt gọi gần đây theo model
        self._calls_lock = threading.Lock()
        self._no_thinking: set[str] = set()     # model không nhận cấu hình thinking
        self._dead_models: set[str] = set()     # model bị 404 (ngừng cấp) trong phiên này
        self._client = None
        self._chat = None
        self._chat_conv_id = None
        self._chat_model = ""
        self._chat_lock = threading.Lock()

    def configure(self, api_key: str, model_name: str) -> None:
        if genai is None:
            raise RuntimeError(
                "Thiếu thư viện google-genai. Chạy: pip install google-genai"
            )
        if not api_key:
            raise RuntimeError("Chưa có Gemini API Key. Vào tab Cài đặt để nhập.")
        model_name = normalize_model(model_name)
        if api_key != self.api_key or model_name != self.model_name or self._client is None:
            self._client = genai.Client(api_key=api_key)
            self._chat = None
            self.api_key = api_key
            self.model_name = model_name

    def _config(self, model: str | None = None):
        """Chế độ nhanh: giảm "thời gian suy nghĩ" của model (thinking) -> phản hồi nhanh hơn rõ rệt."""
        if types is None:
            return None
        model = model or self.model_name
        if self.speed_mode == "fast" and model not in self._no_thinking:
            for kwargs in ({"thinking_level": "low"}, {"thinking_budget": 512}):
                try:
                    return _gen_config(thinking_config=types.ThinkingConfig(**kwargs))
                except Exception:
                    continue
        return _gen_config()

    def _model_rpm(self, model: str) -> int:
        """Giới hạn lượt/phút của gói miễn phí: bản Lite nhận gấp đôi."""
        return 0 if self.rpm_limit <= 0 else self.rpm_limit * (2 if "lite" in model else 1)

    def _rpm_wait(self, model: str) -> float:
        """Số giây phải chờ để không vượt lượt/phút của model (0 = gọi được ngay)."""
        limit = self._model_rpm(model)
        if limit <= 0:
            return 0.0
        now = time.time()
        with self._calls_lock:
            recent = sorted(t for t in self._calls.get(model, []) if now - t < 60.0)
            self._calls[model] = recent
            if len(recent) < limit:
                return 0.0
            return max(0.0, recent[-limit] + 60.0 - now) + 0.3

    def _note_call(self, model: str) -> None:
        with self._calls_lock:
            self._calls.setdefault(model, []).append(time.time())

    def _wait_for_slot(self, model: str, later: list[str],
                       on_retry: Callable[[float, int], None] | None) -> bool:
        """Giữ nhịp lượt/phút. True = gọi model này; False = nên bỏ qua để dùng model dự phòng đang rảnh."""
        wait = self._rpm_wait(model)
        if wait <= 0:
            return True
        if any(self._rpm_wait(m) <= 0 for m in later):
            return False
        if on_retry is not None:
            try:
                on_retry(wait, 0)
            except Exception:
                pass
        time.sleep(min(wait, 65.0))
        return True

    def _candidate_models(self) -> list[str]:
        """Model chính trước, rồi các model dự phòng còn lượt. Nếu tất cả đã hết lượt vẫn thử model chính (có thể đã reset)."""
        order = [self.model_name] + [m for m in self.fallbacks if m != self.model_name]
        alive = [m for m in order if m not in self._dead_models] or order
        usable = [m for m in alive if not (self.quota and self.quota.is_exhausted(m))]
        return usable or alive[:1]

    def _notify_switch(self, old: str, new: str, daily: bool) -> None:
        if self.on_switch is not None:
            try:
                self.on_switch(old, new, daily)
            except Exception:
                pass

    def _generate(self, contents: list, *, retries: int = 2,
                  on_retry: Callable[[float, int], None] | None = None):
        """Gọi Gemini; 429 theo ngày -> đổi sang model dự phòng ngay, 429 theo phút/quá tải -> chờ rồi thử lại."""
        if self._client is None:
            raise RuntimeError("Gemini chưa được cấu hình.")
        if not check_internet():
            raise NetworkError("Lỗi kết nối mạng, vui lòng kiểm tra lại internet rồi thử lại.")
        models = self._candidate_models()
        tried_daily: list[str] = []
        last_exc: Exception | None = None
        for idx, model in enumerate(models):
            delay = 5.0
            attempt = 0
            if not self._wait_for_slot(model, models[idx + 1:], on_retry):
                continue                                   # model này vừa dùng hết lượt/phút -> sang model rảnh
            while True:
                try:
                    self._note_call(model)
                    resp = self._client.models.generate_content(
                        model=model, config=self._config(model), contents=contents)
                    self.last_model = model
                    if self.quota:
                        self.quota.record(model)
                    return resp
                except Exception as exc:  # noqa: BLE001 - mọi lỗi SDK đều xử lý tại đây
                    last_exc = exc
                    if _is_network_error(exc):
                        raise NetworkError(
                            "Lỗi kết nối mạng, vui lòng kiểm tra lại internet rồi thử lại.") from exc
                    text = _err_text(exc)
                    if "thinking" in text and model not in self._no_thinking and "400" in text:
                        self._no_thinking.add(model)      # model này không hỗ trợ thinking -> thử lại không có
                        continue
                    if _is_model_unavailable(exc):
                        self._dead_models.add(model)       # model đã bị ngừng -> không gọi lại nữa
                        if idx < len(models) - 1:
                            break                          # sang model dự phòng
                        raise ApiBusyError(friendly_api_error(exc)) from exc
                    limited = _is_rate_limited(exc)
                    if limited and _is_daily_quota(exc):
                        if self.quota:
                            self.quota.mark_exhausted(model)
                        tried_daily.append(model)
                        break                              # sang model dự phòng
                    transient = limited or _is_overloaded(exc)
                    if transient and attempt < retries:
                        wait = _retry_delay_from(exc) or delay
                        if on_retry is not None:
                            try:
                                on_retry(wait, attempt + 1)
                            except Exception:
                                pass
                        time.sleep(wait)
                        delay = min(60.0, delay * 2)
                        attempt += 1
                        continue
                    if transient and idx < len(models) - 1:
                        break                              # model này đang nghẽn -> thử model khác
                    if transient:
                        raise ApiBusyError(friendly_api_error(exc)) from exc
                    raise
            if idx < len(models) - 1:
                self._notify_switch(model, models[idx + 1], model in tried_daily)
        if tried_daily and len(tried_daily) == len(models):
            raise QuotaExhaustedError(quota_exhausted_message(tried_daily)) from last_exc
        raise ApiBusyError(friendly_api_error(last_exc) if last_exc else "Không gọi được Gemini.")

    def solve_image(self, image: Image.Image, listen_text: str = "",
                    audio_clips: list[bytes] | None = None,
                    on_retry: Callable[[float, int], None] | None = None) -> str:
        """MỘT lần gọi API duy nhất: ảnh câu hỏi + toàn bộ buffer bài nghe (text và/hoặc audio)."""
        prompt = PROMPT_SOLVE
        clips: list[bytes] = []  # KHÔNG BAO GIỜ gửi audio lên API (tiết kiệm token)
        listen_text = listen_text.strip()[-MAX_LISTEN_CHARS:]
        if listen_text:
            prompt += PROMPT_LISTEN_CONTEXT.format(transcript=listen_text)
        if clips:
            prompt += PROMPT_LISTEN_AUDIO
        contents: list[Any] = [
            types.Part.from_bytes(data=_compact_image_bytes(image), mime_type="image/jpeg"),
        ]
        for clip in clips:
            contents.append(types.Part.from_bytes(data=clip, mime_type="audio/wav"))
        contents.append(prompt)
        resp = self._generate(contents, on_retry=on_retry)
        return (getattr(resp, "text", "") or "").strip() or "(Gemini không trả về nội dung)"

    def transcribe_audio(self, wav_bytes: bytes, language: str = "auto") -> str:
        """Chỉ chuyển giọng nói thành văn bản (không trả lời)."""
        if self._client is None:
            raise RuntimeError("Gemini chưa được cấu hình.")
        prompt = PROMPT_TRANSCRIBE.format(lang_hint=LANG_HINT.get(language, LANG_HINT["auto"]))
        resp = self._generate([types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), prompt])
        text = (getattr(resp, "text", "") or "").strip()
        if not text or "KHÔNG CÓ LỜI NÓI" in text.upper():
            return ""
        return text

    def solve_audio(self, wav_bytes: bytes, language: str = "auto") -> str:
        if self._client is None:
            raise RuntimeError("Gemini chưa được cấu hình.")
        prompt = PROMPT_AUDIO.format(lang_hint=LANG_HINT.get(language, LANG_HINT["auto"]))
        resp = self._generate([types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), prompt])
        return (getattr(resp, "text", "") or "").strip() or "(Gemini không trả về nội dung)"

    def solve_text(self, content: str) -> str:
        if self._client is None:
            raise RuntimeError("Gemini chưa được cấu hình.")
        resp = self._generate([PROMPT_TEXT.format(content=content)])
        return (getattr(resp, "text", "") or "").strip() or "(Gemini không trả về nội dung)"

    def chat(self, message: str, conv_id: str | None = None,
             seed: list[tuple[str, str]] | None = None,
             on_text: Callable[[str], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> str:
        """Gửi 1 tin trong cuộc trò chuyện conv_id (streaming). Hết lượt model này thì tự chuyển model dự phòng."""
        if self._client is None:
            raise RuntimeError("Gemini chưa được cấu hình.")
        models = self._candidate_models()
        tried_daily: list[str] = []
        last_exc: Exception | None = None
        for idx, model in enumerate(models):
            with self._chat_lock:
                if self._chat is None or self._chat_conv_id != conv_id or self._chat_model != model:
                    history = [types.Content(role=role, parts=[types.Part(text=text)])
                               for role, text in (seed or [])]
                    self._chat = self._client.chats.create(
                        model=model, config=self._config(model), history=history)
                    self._chat_conv_id = conv_id
                    self._chat_model = model
                chat = self._chat
            parts: list[str] = []
            if idx < len(models) - 1 and not self._wait_for_slot(model, models[idx + 1:], None):
                continue
            if idx == len(models) - 1:
                self._wait_for_slot(model, [], None)
            try:
                self._note_call(model)
                stream = chat.send_message_stream(message)
                for chunk in stream:
                    if should_stop is not None and should_stop():
                        try:
                            stream.close()
                        except Exception:
                            pass
                        break
                    piece = getattr(chunk, "text", "") or ""
                    if piece:
                        parts.append(piece)
                        if on_text is not None:
                            try:
                                on_text("".join(parts))
                            except Exception:
                                pass
                self.last_model = model
                if self.quota:
                    self.quota.record(model)
                return "".join(parts).strip() or "(Gemini không trả về nội dung)"
            except Exception as exc:
                last_exc = exc
                if _is_network_error(exc):
                    raise NetworkError("Lỗi kết nối mạng, vui lòng kiểm tra lại internet rồi thử lại.") from exc
                text = _err_text(exc)
                if "thinking" in text and "400" in text and model not in self._no_thinking and not parts:
                    self._no_thinking.add(model)
                    self._chat = None
                    return self.chat(message, conv_id, seed, on_text, should_stop)
                if _is_model_unavailable(exc):
                    self._dead_models.add(model)
                    with self._chat_lock:
                        self._chat = None
                    if not parts and idx < len(models) - 1:
                        self._notify_switch(model, models[idx + 1], False)
                        continue
                    raise ApiBusyError(friendly_api_error(exc)) from exc
                limited = _is_rate_limited(exc)
                if (limited or _is_overloaded(exc)) and not parts and idx < len(models) - 1:
                    daily = limited and _is_daily_quota(exc)
                    if daily:
                        tried_daily.append(model)
                        if self.quota:
                            self.quota.mark_exhausted(model)
                    with self._chat_lock:
                        self._chat = None
                    self._notify_switch(model, models[idx + 1], daily)
                    continue
                if limited and _is_daily_quota(exc):
                    tried_daily.append(model)
                    if self.quota:
                        self.quota.mark_exhausted(model)
                    if len(tried_daily) == len(models):
                        raise QuotaExhaustedError(quota_exhausted_message(tried_daily)) from exc
                if limited or _is_overloaded(exc):
                    raise ApiBusyError(friendly_api_error(exc)) from exc
                raise
        raise ApiBusyError(friendly_api_error(last_exc) if last_exc else "Không gọi được Gemini.")

    def reset_chat(self) -> None:
        with self._chat_lock:
            self._chat = None
            self._chat_conv_id = None
            self._chat_model = ""


# --------------------------------------------------------------------------- #
# Quản lý phím tắt toàn cục
# --------------------------------------------------------------------------- #
MOD_NAMES = ("ctrl", "shift", "alt", "cmd")


def _norm_key(key) -> str | None:
    if isinstance(key, keyboard.Key):
        n = key.name or ""
        if n == "alt_gr":
            return "alt"
        for base in MOD_NAMES:
            if n.startswith(base):
                return base
        return n
    if isinstance(key, keyboard.KeyCode):
        vk = getattr(key, "vk", None)
        if vk is not None and (65 <= vk <= 90 or 48 <= vk <= 57):
            return chr(vk).lower()
        if vk is not None and 96 <= vk <= 105:
            return str(vk - 96)
        ch = key.char
        if ch:
            if len(ch) == 1 and ord(ch) < 32:
                return chr(ord(ch) + 96)
            return ch.lower()
    return None


def parse_hotkey(text: str) -> frozenset[str]:
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    keys: set[str] = set()
    for p in parts:
        name = p.strip("<>")
        if name in ("control", "ctrl_l", "ctrl_r"):
            name = "ctrl"
        if name in ("win", "super", "command", "cmd_l", "cmd_r"):
            name = "cmd"
        if name in ("alt_l", "alt_r", "alt_gr", "option"):
            name = "alt"
        if name in ("shift_l", "shift_r"):
            name = "shift"
        keys.add(name)
    if not keys or all(k in MOD_NAMES for k in keys):
        raise ValueError("Phím tắt phải có ít nhất một phím thường, vd <ctrl>+<shift>+x")
    return frozenset(keys)


class HotkeyService:
    def __init__(self, on_trigger: Callable[[], None]) -> None:
        self._on_trigger = on_trigger
        self._listener: keyboard.Listener | None = None
        self._target: frozenset[str] = frozenset()
        self._pressed: set[str] = set()
        self._fired = False
        self.active = False

    def start(self, hotkey: str) -> None:
        self.stop()
        self._target = parse_hotkey(hotkey)
        self._pressed.clear()
        self._fired = False
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self._listener.daemon = True
        self._listener.start()
        self._listener.wait()
        self.active = True

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self._pressed.clear()
        self.active = False

    def _press(self, key) -> None:
        name = _norm_key(key)
        if not name:
            return
        self._pressed.add(name)
        if not self._fired and self._target and self._target <= self._pressed:
            self._fired = True
            try:
                self._on_trigger()
            except Exception as exc:
                print("Hotkey error:", exc)

    def _release(self, key) -> None:
        name = _norm_key(key)
        if name:
            self._pressed.discard(name)
        if not (self._target <= self._pressed):
            self._fired = False


MOUSE_BUTTONS = {"right": "Chuột phải", "middle": "Chuột giữa", "x1": "Nút bên X1", "x2": "Nút bên X2"}
MOUSE_CLICKS = {1: "1 lần", 2: "2 lần (nhấp đúp)"}


def _double_click_interval() -> float:
    """Khoảng thời gian tối đa giữa 2 lần nhấp để tính là nhấp đúp (lấy theo cài đặt Windows)."""
    try:
        if sys.platform == "win32":
            import ctypes
            return min(0.8, max(0.3, ctypes.windll.user32.GetDoubleClickTime() / 1000.0))
    except Exception:
        pass
    return 0.5


class MouseTriggerService:
    """Lắng nghe nút chuột toàn cục (pynput) để kích hoạt chụp; có chống bấm liên tiếp."""

    def __init__(self, on_trigger: Callable[[], None], cooldown: float = 1.5) -> None:
        self._on_trigger = on_trigger
        self._listener: mouse.Listener | None = None
        self._button = "right"
        self._clicks = 1
        self._presses: list[float] = []
        self._cooldown = cooldown
        self._last = 0.0
        self.active = False

    def start(self, button: str, clicks: int = 1) -> None:
        self.stop()
        self._button = button if button in MOUSE_BUTTONS else "right"
        self._clicks = 2 if int(clicks or 1) >= 2 else 1
        self._presses = []
        self._listener = mouse.Listener(on_click=self._click)
        self._listener.daemon = True
        self._listener.start()
        self.active = True

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self.active = False

    def _click(self, _x, _y, button, pressed) -> None:
        if not pressed:
            return
        target = getattr(mouse.Button, self._button, None)
        if target is None or button != target:
            return
        import time as _time
        now = _time.monotonic()
        if now - self._last < self._cooldown:
            return
        if self._clicks > 1:
            window = _double_click_interval()
            self._presses = [t for t in self._presses if now - t <= window]
            self._presses.append(now)
            if len(self._presses) < self._clicks:
                return
            self._presses.clear()
        self._last = now
        try:
            self._on_trigger()
        except Exception as exc:
            print("Mouse trigger error:", exc)


def _shot_time(path: Path) -> _dt.datetime:
    m = re.search(r"(\d{8})_(\d{6})", path.stem)
    if m:
        try:
            return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return _dt.datetime.fromtimestamp(path.stat().st_mtime)


def list_screenshots() -> list[tuple[Path, _dt.datetime]]:
    files = [p for p in SHOTS_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(((p, _shot_time(p)) for p in files), key=lambda t: t[1], reverse=True)


def _parse_hhmm(text: str) -> _dt.time:
    return _dt.datetime.strptime(text.strip(), "%H:%M").time()


def shots_in_range(shots, day: str, start: str, end: str):
    d = _dt.date.fromisoformat(day.strip())
    t1, t2 = _parse_hhmm(start), _parse_hhmm(end)
    if t2 < t1:
        raise ValueError("Giờ kết thúc phải sau giờ bắt đầu")
    return [(p, t) for p, t in shots if t.date() == d and t1 <= t.time().replace(second=0) <= t2]


class HotkeyRecorder:
    def __init__(self, on_done: Callable[[str], None]) -> None:
        self._on_done = on_done
        self._mods: set[str] = set()
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        self.stop()
        self._mods.clear()
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _press(self, key) -> None:
        name = _norm_key(key)
        if not name:
            return
        if name in MOD_NAMES:
            self._mods.add(name)
            return
        main = name if len(name) == 1 else f"<{name}>"
        order = [m for m in ("ctrl", "alt", "shift", "cmd") if m in self._mods]
        combo = "+".join([f"<{m}>" for m in order] + [main])
        self.stop()
        self._on_done(combo)

    def _release(self, key) -> None:
        name = _norm_key(key)
        if name in MOD_NAMES:
            self._mods.discard(name)


# --------------------------------------------------------------------------- #
# Chọn vùng màn hình & hiệu ứng
# --------------------------------------------------------------------------- #
def grab_virtual_screen() -> tuple[Image.Image, dict[str, int]]:
    with _new_mss() as sct:
        mon = dict(sct.monitors[0])
        raw = sct.grab(mon)
        return Image.frombytes("RGB", raw.size, raw.rgb), mon


def grab_screen(region: list[int] | None) -> Image.Image:
    with _new_mss() as sct:
        if region and region[2] > 0 and region[3] > 0:
            box = {"left": int(region[0]), "top": int(region[1]),
                   "width": int(region[2]), "height": int(region[3])}
        else:
            box = sct.monitors[1]
        raw = sct.grab(box)
        return Image.frombytes("RGB", raw.size, raw.rgb)


class RegionSelector:
    def __init__(self, root: tk.Misc,
                 on_done: Callable[[list[int] | None, Image.Image | None], None]) -> None:
        self.on_done = on_done
        self.full, self.mon = grab_virtual_screen()
        W, H = self.full.size
        dim = ImageEnhance.Brightness(self.full).enhance(0.45)

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{W}x{H}+{self.mon['left']}+{self.mon['top']}")
        self.win.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.win, width=W, height=H, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._bg = ImageTk.PhotoImage(dim)
        self.canvas.create_image(0, 0, image=self._bg, anchor="nw")
        self._crop_img = None
        self._crop_id = None
        self.rect = None
        self.size_txt = None
        self.start: tuple[int, int] | None = None

        cx = -self.mon["left"] + 960 if W > 1920 else W // 2
        self.hint = self.canvas.create_text(
            cx, 48, text="Kéo chuột để chọn vùng câu hỏi   •   Chuột phải / ESC để huỷ",
            fill="#ffffff", font=(FONT_FAMILY, 16, "bold"))

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-3>", lambda _e: self._finish(None))
        self.win.bind("<Escape>", lambda _e: self._finish(None))
        safe_after(self.win, 10, self._grab_focus)

    def _grab_focus(self) -> None:
        try:
            self.win.lift()
            self.win.focus_force()
            self.win.grab_set()
        except Exception:
            pass

    def _press(self, e) -> None:
        self.start = (e.x, e.y)
        self.canvas.itemconfigure(self.hint, state="hidden")
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                 outline="#3b82f6", width=2)
        self.size_txt = self.canvas.create_text(e.x, e.y - 12, text="", fill="#3b82f6",
                                                anchor="sw", font=(FONT_FAMILY, 11, "bold"))

    def _drag(self, e) -> None:
        if not self.start:
            return
        x0, y0 = self.start
        l, t, r, b = min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y)
        if r - l > 2 and b - t > 2:
            self._crop_img = ImageTk.PhotoImage(self.full.crop((l, t, r, b)))
            if self._crop_id is None:
                self._crop_id = self.canvas.create_image(l, t, image=self._crop_img, anchor="nw")
            else:
                self.canvas.itemconfigure(self._crop_id, image=self._crop_img)
                self.canvas.coords(self._crop_id, l, t)
            self.canvas.tag_raise(self.rect)
            self.canvas.tag_raise(self.size_txt)
        self.canvas.coords(self.rect, l, t, r, b)
        self.canvas.coords(self.size_txt, l, max(t - 6, 14))
        self.canvas.itemconfigure(self.size_txt, text=f"{r - l} × {b - t}")

    def _release(self, e) -> None:
        if not self.start:
            return self._finish(None)
        x0, y0 = self.start
        l, t, r, b = min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y)
        if r - l < 12 or b - t < 12:
            return self._finish(None)
        img = self.full.crop((l, t, r, b))
        region = [l + self.mon["left"], t + self.mon["top"], r - l, b - t]
        self._finish(region, img)

    def _finish(self, region, img=None) -> None:
        try:
            self.win.grab_release()
            self.win.destroy()
        except Exception:
            pass
        self.on_done(region, img)


def flash_screen(root: tk.Misc, region: list[int] | None = None) -> None:
    with _new_mss() as sct:
        m = sct.monitors[1]
    if region:
        x, y, w, h = region
    else:
        x, y, w, h = m["left"], m["top"], m["width"], m["height"]
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.configure(bg="white")
    win.geometry(f"{w}x{h}+{x}+{y}")
    steps = [0.75, 0.55, 0.38, 0.22, 0.1, 0.0]

    def tick(i: int = 0) -> None:
        if i >= len(steps):
            win.destroy()
            return
        try:
            win.attributes("-alpha", steps[i])
        except Exception:
            pass
        safe_after(win, 35, tick, i + 1)

    tick()


class ScreenToast:
    def __init__(self, root: tk.Misc, on_click: Callable[[], None]) -> None:
        self.root = root
        self.on_click = on_click
        self.win: tk.Toplevel | None = None
        self._hide_job = None

    def show(self, title: str, body: str = "", color: str = "#3b82f6",
             timeout_ms: int | None = 3500) -> None:
        if self.win is None or not self.win.winfo_exists():
            self.win = tk.Toplevel(self.root)
            self.win.overrideredirect(True)
            self.win.attributes("-topmost", True)
            try:
                self.win.attributes("-alpha", 0.97)
            except Exception:
                pass
            self.win.configure(bg="#18181b")
            self.bar = tk.Frame(self.win, bg=color, width=3)
            self.bar.pack(side="left", fill="y")
            box = tk.Frame(self.win, bg="#18181b", padx=11, pady=8)
            box.pack(side="left", fill="both", expand=True)
            self.t_lbl = tk.Label(box, bg="#18181b", fg="#f4f4f5", anchor="w",
                                  justify="left", wraplength=250, font=(FONT_FAMILY, 10, "bold"))
            self.t_lbl.pack(fill="x")
            self.b_lbl = tk.Label(box, bg="#18181b", fg="#a1a1aa", anchor="w",
                                  justify="left", wraplength=250, font=(FONT_FAMILY, 9))
            self.b_lbl.pack(fill="x", pady=(2, 0))
            for w in (self.win, box, self.t_lbl, self.b_lbl):
                w.bind("<Button-1>", lambda _e: self._clicked())
        self.bar.configure(bg=color)
        # Bản đầy đủ vẫn nằm trong ứng dụng; thông báo chỉ hiển thị tóm tắt.
        short_title = title if len(title) <= 52 else title[:51].rstrip() + "…"
        short_body = " ".join(body.split())
        if len(short_body) > 88:
            short_body = short_body[:87].rstrip() + "…"
        self.t_lbl.configure(text=short_title)
        self.b_lbl.configure(text=short_body)
        self.b_lbl.pack_forget() if not short_body else self.b_lbl.pack(fill="x", pady=(2, 0))
        self.win.update_idletasks()
        w, h = min(self.win.winfo_reqwidth(), 280), self.win.winfo_reqheight()
        with _new_mss() as sct:
            m = sct.monitors[1]
        x = m["left"] + m["width"] - w - 16
        y = m["top"] + 18
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        self.win.lift()
        if self._hide_job:
            safe_after_cancel(self.root, self._hide_job)
            self._hide_job = None
        if timeout_ms:
            self._hide_job = safe_after(self.root, timeout_ms, self.hide)

    def _clicked(self) -> None:
        self.hide()
        self.on_click()

    def hide(self) -> None:
        self._hide_job = None
        if self.win is not None:
            try:
                self.win.destroy()
            except Exception:
                pass
            self.win = None


# --------------------------------------------------------------------------- #
# Chế độ nghe audio: thu âm từ micro hoặc loopback (Stereo Mix / VB-Cable)
# --------------------------------------------------------------------------- #
def audio_input_devices() -> list[str]:
    """Danh sách thiết bị thu âm; bao gồm cả thiết bị loopback nếu hệ thống có."""
    if sd is None:
        return []
    names: list[str] = []
    try:
        for dev in sd.query_devices():
            if int(dev.get("max_input_channels", 0)) > 0:
                name = str(dev.get("name", "")).strip()
                if name and name not in names:
                    names.append(name)
    except Exception:
        return []
    return names


def _device_index(name: str) -> int | None:
    if sd is None or not name:
        return None
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) > 0 and str(dev.get("name", "")) == name:
                return idx
    except Exception:
        return None
    return None


def pcm_to_wav(frames: list, samplerate: int = AUDIO_SAMPLERATE) -> bytes:
    """Ghép các block float32 mono thành file WAV 16-bit trong bộ nhớ."""
    if np is None or not frames:
        return b""
    data = np.concatenate(frames).astype("float32")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = data / max(peak, 1e-6) * 0.95
    pcm = (data * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm)
    return buf.getvalue()


class AudioListener:
    """Nghe liên tục, tự cắt đoạn theo khoảng lặng (VAD đơn giản theo RMS)."""

    BLOCK = 1600  # 0.1 giây @16kHz

    def __init__(self,
                 on_segment: Callable[[bytes, float], None],
                 on_level: Callable[[float, bool], None] | None = None,
                 on_auto_stop: Callable[[], None] | None = None) -> None:
        self.on_segment = on_segment
        self.on_level = on_level
        self.on_auto_stop = on_auto_stop
        self.active = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.device_name = ""
        self.threshold = float(DEFAULT_CONFIG["listen_threshold"])
        self.silence_sec = float(DEFAULT_CONFIG["listen_silence"])
        self.min_speech = float(DEFAULT_CONFIG["listen_min_speech"])
        # Tự tắt ghi âm nếu không nghe thấy âm thanh nào trong khoảng này
        self.auto_stop_sec = float(DEFAULT_CONFIG.get("listen_auto_stop", 15))
        self.last_error = ""

    # -- điều khiển ------------------------------------------------------- #
    def start(self, device_name: str = "", threshold: float | None = None,
              silence_sec: float | None = None, min_speech: float | None = None) -> None:
        if sd is None or np is None:
            raise RuntimeError(
                "Thiếu thư viện âm thanh. Chạy: pip install sounddevice numpy"
            )
        if self.active:
            return
        self.device_name = device_name or ""
        if threshold is not None:
            self.threshold = float(threshold)
        if silence_sec is not None:
            self.silence_sec = float(silence_sec)
        if min_speech is not None:
            self.min_speech = float(min_speech)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.active = True

    def stop(self) -> None:
        self._stop.set()
        self.active = False
        thread, self._thread = self._thread, None
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    # -- vòng lặp thu âm --------------------------------------------------- #
    def _run(self) -> None:
        audio_q: "queue.Queue[Any]" = queue.Queue()

        def callback(indata, _frames, _time_info, _status):
            audio_q.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=AUDIO_SAMPLERATE, channels=1, dtype="float32",
                blocksize=self.BLOCK, device=_device_index(self.device_name),
                callback=callback,
            )
        except Exception as exc:
            if self.on_level:
                self.on_level(-1.0, False)
            print("Không mở được thiết bị âm thanh:", exc)
            self.last_error = str(exc)
            self.active = False
            return

        buffer: list = []
        speaking = False
        silence_blocks = 0
        block_sec = self.BLOCK / AUDIO_SAMPLERATE
        need_silence = max(1, int(self.silence_sec / block_sec))
        last_loud = time.monotonic()
        auto_stopped = False

        with stream:
            while not self._stop.is_set():
                try:
                    block = audio_q.get(timeout=0.3)
                except queue.Empty:
                    block = None
                # Không nghe thấy âm thanh nào quá lâu -> tự tắt ghi âm
                if self.auto_stop_sec > 0 and time.monotonic() - last_loud > self.auto_stop_sec:
                    auto_stopped = True
                    break
                if block is None:
                    continue
                level = float(np.sqrt(np.mean(np.square(block))))
                loud = level >= self.threshold
                if loud:
                    last_loud = time.monotonic()
                if self.on_level:
                    self.on_level(level, speaking)

                if loud:
                    speaking = True
                    silence_blocks = 0
                    buffer.append(block)
                elif speaking:
                    buffer.append(block)
                    silence_blocks += 1
                    if silence_blocks >= need_silence:
                        duration = len(buffer) * block_sec
                        segment, buffer = buffer, []
                        speaking = False
                        silence_blocks = 0
                        if duration >= self.min_speech:
                            wav = pcm_to_wav(segment)
                            if wav:
                                self.on_segment(wav, duration)
                # nói liên tục quá 15 giây thì chốt đoạn để chép lời gần thời gian thực
                if len(buffer) * block_sec > 15:
                    wav = pcm_to_wav(buffer)
                    buffer, speaking, silence_blocks = [], False, 0
                    if wav:
                        self.on_segment(wav, 60.0)
        self.active = False
        if auto_stopped and self.on_auto_stop:
            try:
                self.on_auto_stop()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Làm đẹp văn bản: LaTeX -> Unicode toán học
# --------------------------------------------------------------------------- #
_SUP_MAP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
    "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻",
    "=": "⁼", "(": "⁽", ")": "⁾", " ": " ",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
    "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ",
    "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
    "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
    "z": "ᶻ",
}

_SUB_MAP = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅",
    "6": "₆", "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋",
    "=": "₌", "(": "₍", ")": "₎", " ": " ",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
    "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

_LATEX_SYMBOLS: list[tuple[str, str]] = [
    (r"\mathbb{R}", "ℝ"), (r"\mathbb{N}", "ℕ"), (r"\mathbb{Z}", "ℤ"),
    (r"\mathbb{Q}", "ℚ"), (r"\mathbb{C}", "ℂ"),
    (r"\infty", "∞"), (r"\partial", "∂"), (r"\nabla", "∇"),
    (r"\int", "∫"), (r"\iint", "∬"), (r"\oint", "∮"), (r"\sum", "∑"), (r"\prod", "∏"),
    (r"\pm", "±"), (r"\mp", "∓"), (r"\times", "×"), (r"\div", "÷"), (r"\cdot", "·"),
    (r"\leq", "≤"), (r"\le", "≤"), (r"\geq", "≥"), (r"\ge", "≥"),
    (r"\neq", "≠"), (r"\ne", "≠"), (r"\approx", "≈"), (r"\equiv", "≡"), (r"\sim", "∼"),
    (r"\rightarrow", "→"), (r"\Rightarrow", "⇒"), (r"\leftarrow", "←"),
    (r"\Leftarrow", "⇐"), (r"\leftrightarrow", "↔"), (r"\to", "→"), (r"\mapsto", "↦"),
    (r"\forall", "∀"), (r"\exists", "∃"), (r"\nexists", "∄"),
    (r"\in", "∈"), (r"\notin", "∉"), (r"\subseteq", "⊆"), (r"\subset", "⊂"),
    (r"\supseteq", "⊇"), (r"\supset", "⊃"), (r"\cup", "∪"), (r"\cap", "∩"),
    (r"\emptyset", "∅"), (r"\varnothing", "∅"), (r"\angle", "∠"), (r"\perp", "⊥"),
    (r"\parallel", "∥"), (r"\degree", "°"), (r"\circ", "∘"), (r"\ldots", "…"),
    (r"\dots", "…"), (r"\cdots", "⋯"), (r"\prime", "′"), (r"\therefore", "∴"),
    (r"\alpha", "α"), (r"\beta", "β"), (r"\gamma", "γ"), (r"\delta", "δ"),
    (r"\Delta", "Δ"), (r"\epsilon", "ε"), (r"\varepsilon", "ε"), (r"\zeta", "ζ"),
    (r"\eta", "η"), (r"\theta", "θ"), (r"\Theta", "Θ"), (r"\lambda", "λ"),
    (r"\Lambda", "Λ"), (r"\mu", "μ"), (r"\nu", "ν"), (r"\xi", "ξ"), (r"\pi", "π"),
    (r"\Pi", "Π"), (r"\rho", "ρ"), (r"\sigma", "σ"), (r"\Sigma", "Σ"),
    (r"\tau", "τ"), (r"\phi", "φ"), (r"\varphi", "φ"), (r"\Phi", "Φ"),
    (r"\chi", "χ"), (r"\psi", "ψ"), (r"\Psi", "Ψ"), (r"\omega", "ω"), (r"\Omega", "Ω"),
]

_BRACED = r"\{([^{}]*)\}"


def _strip_math_delims(text: str) -> str:
    text = re.sub(r"\\\[(.+?)\\\]", r" \1 ", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text, flags=re.S)
    text = re.sub(r"\$\$(.+?)\$\$", r" \1 ", text, flags=re.S)
    text = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", r"\1", text, flags=re.S)
    return text.replace("\\$", "$")


def _to_script(body: str, table: dict[str, str]) -> str | None:
    """Chuyển toàn bộ nội dung sang ký tự trên/dưới; None nếu có ký tự không hỗ trợ."""
    body = body.strip()
    if not body:
        return None
    out = []
    for ch in body:
        mapped = table.get(ch)
        if mapped is None:
            return None
        out.append(mapped)
    return "".join(out)


def latex_to_unicode(text: str) -> str:
    """Chuyển các ký hiệu LaTeX phổ biến sang Unicode để hiển thị trong Tkinter."""
    if not text:
        return ""
    out = _strip_math_delims(text)

    out = re.sub(r"\\sqrt\s*\[(.*?)\]\s*" + _BRACED, lambda m: f"({m.group(1)})√({m.group(2)})", out)
    out = re.sub(r"\\sqrt\s*" + _BRACED, lambda m: f"√({m.group(1)})", out)
    out = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*" + _BRACED, r"\1", out)

    # \frac{a}{b} -> (a)/(b) ; lặp nhiều vòng để xử lý phân số lồng nhau
    for _ in range(5):
        new = re.sub(r"\\(?:d|t)?frac\s*" + _BRACED + r"\s*" + _BRACED,
                     lambda m: f"({m.group(1)})/({m.group(2)})", out)
        if new == out:
            break
        out = new
    out = re.sub(r"\\sqrt\s*" + _BRACED, lambda m: f"√({m.group(1)})", out)
    out = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg|displaystyle|limits|quad|qquad)\b", " ", out)
    out = re.sub(r"\\[!,;:]", " ", out)
    out = out.replace("\\ ", " ").replace("\\\\", "\n")

    out = re.sub(r"\^\s*\\circ", "°", out)
    out = re.sub(r"\^\s*\{\s*\\circ\s*\}", "°", out)

    for src, dst in _LATEX_SYMBOLS:
        out = re.sub(re.escape(src) + r"(?![A-Za-z])", dst, out)

    # mũ và chỉ số
    def _sup(m):
        body = m.group(1) or m.group(2) or ""
        return _to_script(body, _SUP_MAP) or f"^({body})"

    def _subf(m):
        body = m.group(1) or m.group(2) or ""
        return _to_script(body, _SUB_MAP) or f"_({body})"

    out = re.sub(r"\^\{([^{}]*)\}|\^([A-Za-z0-9+\-])", _sup, out)
    out = re.sub(r"_\{([^{}]*)\}|_([A-Za-z0-9+\-])", _subf, out)

    out = out.replace("{", "").replace("}", "")
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def clean_display_text(text: str) -> str:
    """Bỏ rườm rà markdown + chuyển LaTeX sang Unicode cho giao diện."""
    if not text:
        return ""
    out = latex_to_unicode(str(text))
    out = re.sub(r"`{1,3}", "", out)
    out = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", out, flags=re.S)
    out = re.sub(r"^#{1,6}\s*", "", out, flags=re.M)
    out = re.sub(r"^\s*[-*]\s+", "• ", out, flags=re.M)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# --------------------------------------------------------------------------- #
# Giao diện chính ứng dụng
# --------------------------------------------------------------------------- #
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def _short_answer(answer: str) -> str:
    for line in answer.splitlines():
        if "Đáp án" in line or "Đáp án đúng" in line:
            return clean_display_text(line)[:220]
    return clean_display_text(answer)[:220]


# --------------------------------------------------------------------------- #
# Dữ liệu câu hỏi và trạng thái ôn tập
# --------------------------------------------------------------------------- #
CHOICES = ("A", "B", "C", "D")

PALETTE = {
    "window": "#050516",
    "sidebar": "#070719",
    "surface": "#191F36",
    "surface_2": "#212842",
    "surface_hover": "#2C3555",
    "border": "#363E5E",
    "text": "#DADEF0",
    "muted": "#868EB0",
    "primary": "#3B4BDB",
    "primary_hover": "#3341C2",
    "blue": "#9AA3F0",
    "success": "#27B873",
    "success_soft": "#0B2A1F",
    "danger": "#EE6A80",
    "danger_soft": "#2E1528",
    "warning": "#E8B91C",
}


# ---------------------------------------------------------------------------
# Phông chữ giao diện: Be Vietnam Pro (hỗ trợ đầy đủ dấu tiếng Việt, nét tròn dễ đọc).
# Phông nằm trong thư mục "fonts" cạnh file này; nếu thiếu sẽ thử tải 1 lần, không được thì dùng Segoe UI.
# ---------------------------------------------------------------------------
FONT_DIR = BASE_DIR / "fonts"
_UI_FONT_NAME = "Be Vietnam Pro"
_UI_FONT_FILES = ("BeVietnamPro-Regular.ttf", "BeVietnamPro-Bold.ttf",
                  "BeVietnamPro-Italic.ttf", "BeVietnamPro-BoldItalic.ttf")
_UI_FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/bevietnampro/"


def _setup_ui_font() -> str:
    try:
        FONT_DIR.mkdir(exist_ok=True)
        for name in _UI_FONT_FILES:
            path = FONT_DIR / name
            if path.exists():
                continue
            try:
                import urllib.request
                with urllib.request.urlopen(_UI_FONT_URL + name, timeout=6) as resp:
                    path.write_bytes(resp.read())
            except Exception:
                break  # không có mạng: bỏ qua, dùng phông dự phòng
        loaded = False
        for name in _UI_FONT_FILES:
            path = FONT_DIR / name
            if path.exists():
                ok = ctk.FontManager.load_font(str(path))
                loaded = loaded or (name.endswith("Regular.ttf") and bool(ok))
        if loaded:
            return _UI_FONT_NAME
    except Exception:
        pass
    return "Segoe UI"


FONT_FAMILY = _setup_ui_font()
try:
    ctk.ThemeManager.theme["CTkFont"]["family"] = FONT_FAMILY
except Exception:
    pass


def _apply_theme_colors() -> None:
    """Đồng bộ màu mặc định của customtkinter với bảng màu navy (kiểu Quizlet)."""
    theme = ctk.ThemeManager.theme
    P = PALETTE

    def put(widget: str, **colors: str) -> None:
        try:
            theme[widget].update(colors)
        except Exception:
            pass

    put("CTk", fg_color=P["window"])
    put("CTkToplevel", fg_color=P["window"])
    put("CTkFrame", fg_color=P["surface"], top_fg_color=P["surface_2"], border_color=P["border"])
    put("CTkButton", fg_color=P["primary"], hover_color=P["primary_hover"],
        border_color=P["border"], text_color=P["text"])
    put("CTkLabel", text_color=P["text"])
    put("CTkEntry", fg_color=P["surface"], border_color=P["border"], text_color=P["text"],
        placeholder_text_color=P["muted"])
    put("CTkTextbox", fg_color=P["surface"], border_color=P["border"], text_color=P["text"])
    put("CTkCheckBox", fg_color=P["primary"], hover_color=P["primary_hover"],
        border_color=P["border"], checkmark_color="#FFFFFF", text_color=P["text"])
    put("CTkRadioButton", fg_color=P["primary"], hover_color=P["primary_hover"],
        border_color=P["border"], text_color=P["text"])
    put("CTkOptionMenu", fg_color=P["surface_2"], button_color=P["primary"],
        button_hover_color=P["primary_hover"], text_color=P["text"])
    put("CTkComboBox", fg_color=P["surface"], border_color=P["border"], button_color=P["surface_2"],
        button_hover_color=P["surface_hover"], text_color=P["text"])
    put("CTkProgressBar", fg_color=P["surface_2"], progress_color=P["primary"], border_color=P["border"])
    put("CTkSegmentedButton", fg_color=P["surface"], selected_color=P["primary"],
        selected_hover_color=P["primary_hover"], unselected_color=P["surface"],
        unselected_hover_color=P["surface_hover"], text_color=P["text"])
    put("DropdownMenu", fg_color=P["surface"], hover_color=P["surface_hover"], text_color=P["text"])


_apply_theme_colors()


# ---------------------------------------------------------------------------
# Avatar / icon của ứng dụng (tự vẽ bằng Pillow, không cần file ngoài).
# Muốn dùng ảnh riêng: đặt "icon.png" (hoặc "avatar.png") cạnh file này, app sẽ ưu tiên ảnh đó.
# ---------------------------------------------------------------------------
def build_app_icon(size: int = 256) -> Image.Image:
    S = 512
    top, bottom = (96, 112, 255), (28, 34, 140)
    grad = Image.composite(
        Image.new("RGB", (S, S), bottom), Image.new("RGB", (S, S), top),
        Image.linear_gradient("L").resize((S, S)),
    ).convert("RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=118, fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((84, 96, 428, 376), radius=44, fill=(240, 243, 255, 255))       # thẻ câu hỏi
    d.rounded_rectangle((126, 138, 290, 168), radius=15, fill=(59, 75, 219, 255))       # tiêu đề
    d.rounded_rectangle((126, 204, 388, 240), radius=18, fill=(205, 212, 247, 255))     # đáp án 1
    d.rounded_rectangle((126, 266, 330, 302), radius=18, fill=(205, 212, 247, 255))     # đáp án 2
    cx, cy, r = 372, 372, 98
    d.ellipse((cx - r - 12, cy - r - 12, cx + r + 12, cy + r + 12), fill=(30, 38, 150, 255))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(39, 196, 122, 255))                # huy hiệu xanh
    pts = [(cx - 52, cy + 4), (cx - 14, cy + 42), (cx + 56, cy - 40)]
    d.line(pts, fill=(255, 255, 255, 255), width=28, joint="curve")
    for x, y in (pts[0], pts[-1]):
        d.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(255, 255, 255, 255))
    return img.resize((size, size), Image.LANCZOS)


def load_app_icon(size: int = 256) -> Image.Image:
    for name in ("icon.png", "avatar.png", "icon.jpg", "avatar.jpg"):
        path = BASE_DIR / name
        if path.exists():
            try:
                img = Image.open(path).convert("RGBA")
                side = min(img.size)
                left, top = (img.width - side) // 2, (img.height - side) // 2
                return img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
            except Exception:
                break
    return build_app_icon(size)


if sys.platform == "win32":
    try:  # để thanh taskbar hiện icon của app thay vì icon Python
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StudyQuestionTracker.App")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cuộn chuột mượt: mỗi nấc lăn chuột được "trượt" dần (easing) thay vì nhảy cục.
# Áp dụng cho mọi CTkScrollableFrame và mọi ô văn bản (chat, nhật ký...).
# ---------------------------------------------------------------------------
try:
    ctk.ThemeManager.theme["CTkScrollbar"].update({
        "fg_color": "transparent",
        "button_color": PALETTE["border"],
        "button_hover_color": PALETTE["muted"],
    })
except Exception:
    pass


def _patch_ctk_scrollable_frame() -> None:
    """Tắt cách cuộn mặc định (nhảy từng nấc) của CTkScrollableFrame, SmoothScroller sẽ thay thế."""
    cls = ctk.CTkScrollableFrame
    if getattr(cls, "_smooth_patched", False):
        return
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            self._parent_frame._smooth_owner = self  # để tìm ra khung cuộn khi chuột ở trên thanh cuộn
            self._parent_canvas.configure(yscrollincrement=1)  # cuộn từng pixel thay vì nhảy 30px
        except Exception:
            pass

    cls.__init__ = patched_init
    cls._mouse_wheel_all = lambda self, event=None: None
    cls._smooth_patched = True


_patch_ctk_scrollable_frame()


class SmoothScroller:
    TICK_MS = 12            # ~80 khung hình/giây
    PIXELS_PER_NOTCH = 96   # một nấc lăn chuột ≈ 96 px
    EASE = 0.19             # càng nhỏ càng "lướt" lâu
    MAX_PENDING = 2200      # giới hạn quãng đường chờ cuộn khi lăn rất nhanh

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self._state: dict[Any, dict[str, Any]] = {}
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind_all(seq, self._on_wheel, add="+")
            # Thay binding mặc định (nhảy từng dòng) của ô Text bằng bản mượt.
            root.bind_class("Text", seq, self._on_text_wheel)

    @staticmethod
    def _notches(event: Any) -> float:
        num = getattr(event, "num", None)
        if num == 4:
            return -1.0
        if num == 5:
            return 1.0
        delta = getattr(event, "delta", 0) or 0
        if not delta:
            return 0.0
        return -delta / 3.0 if sys.platform == "darwin" else -delta / 120.0

    def _on_text_wheel(self, event: Any) -> str:
        self._on_wheel(event)
        return "break"

    @staticmethod
    def _region_height(canvas: tk.Canvas) -> float:
        try:
            region = [float(v) for v in str(canvas.cget("scrollregion")).split()]
            return region[3] - region[1]
        except Exception:
            return 0.0

    def _can_scroll(self, widget: Any, sign: int) -> bool:
        try:
            if isinstance(widget, tk.Canvas):
                total = self._region_height(widget)
                max_top = total - widget.winfo_height()
                if max_top <= 1:
                    return False
                top = widget.yview()[0] * total
                return top > 0.5 if sign < 0 else top < max_top - 0.5
            first, last = widget.yview()
            return first > 0.0001 if sign < 0 else last < 0.9999
        except Exception:
            return False

    def _candidates(self, widget: Any):
        seen: set[int] = set()
        w = widget
        while w is not None:
            owner = w if isinstance(w, ctk.CTkScrollableFrame) else getattr(w, "_smooth_owner", None)
            target = None
            if owner is not None:
                if getattr(owner, "_orientation", "vertical") == "vertical":
                    target = getattr(owner, "_parent_canvas", None)
            elif isinstance(w, tk.Text):
                target = w
            if target is not None and id(target) not in seen:
                seen.add(id(target))
                yield target
            w = getattr(w, "master", None)

    def _on_wheel(self, event: Any) -> None:
        notches = self._notches(event)
        if not notches or (getattr(event, "state", 0) & 0x0005):  # Shift/Ctrl + lăn: bỏ qua
            return None
        try:
            under = self.root.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return None
        if under is None:
            return None
        sign = 1 if notches > 0 else -1
        chosen = None
        for cand in self._candidates(under):
            st = self._state.get(cand)
            if (st and st["pending"] * sign > 0) or self._can_scroll(cand, sign):
                chosen = cand
                break
        if chosen is None:
            return None
        st = self._state.setdefault(chosen, {"pending": 0.0, "carry": 0.0, "job": None})
        if st["pending"] * sign < 0:      # đổi chiều lăn -> phản hồi ngay
            st["pending"] = 0.0
            st["carry"] = 0.0
        st["pending"] = max(-self.MAX_PENDING,
                            min(self.MAX_PENDING, st["pending"] + notches * self.PIXELS_PER_NOTCH))
        if st["job"] is None:
            st["job"] = chosen.after(self.TICK_MS, lambda w=chosen: self._tick(w))
        return None

    def _scroll_px(self, widget: Any, px: int) -> bool:
        if isinstance(widget, tk.Canvas):
            total = self._region_height(widget)
            max_top = total - widget.winfo_height()
            if max_top <= 0:
                return False
            top = widget.yview()[0] * total
            new = min(max(top + px, 0.0), max_top)
            if abs(new - top) < 0.5:
                return False
            widget.yview_moveto(new / total)
            return True
        before = widget.yview()
        widget.yview_scroll(px, "pixels")
        return widget.yview() != before

    def _tick(self, widget: Any) -> None:
        st = self._state.get(widget)
        if st is None:
            return
        st["job"] = None
        try:
            pending = st["pending"]
            step = pending * self.EASE
            if abs(step) < 1.0:
                step = 1.0 if pending > 0 else -1.0
                if abs(pending) < 1.0:
                    step = pending
            st["carry"] += step
            whole = int(st["carry"])
            st["carry"] -= whole
            st["pending"] = pending - step
            if whole and not self._scroll_px(widget, whole):
                st["pending"] = 0.0           # chạm đầu/cuối trang thì dừng
            if abs(st["pending"]) >= 0.5:
                st["job"] = widget.after(self.TICK_MS, lambda w=widget: self._tick(w))
            else:
                self._state.pop(widget, None)
        except tk.TclError:
            self._state.pop(widget, None)  # widget đã bị huỷ


# Markdown renderer for the Tk chat transcript (no HTML or extra dependency).
_CHAT_INLINE = re.compile(
    r"(`[^`\n]+`|\*\*\*.+?\*\*\*|___.+?___|\*\*.+?\*\*|__.+?__|"
    r"(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_))"
)
_CHAT_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$")
_CHAT_TABLE_RULE = re.compile(r"^:?-{3,}:?$")


def _chat_cells(line: str) -> list[str]:
    """Split an ordinary Markdown table row, preserving escaped pipes."""
    row = line.strip().strip("|")
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", row)]


def _chat_blocks(source: str) -> list[tuple[str, Any]]:
    """Parse chat Markdown into display blocks without altering the original answer."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[tuple[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("```"):
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code)))
            i += 1
            continue
        if ("|" in line and i + 1 < len(lines)
                and all(_CHAT_TABLE_RULE.fullmatch(cell) for cell in _chat_cells(lines[i + 1]))
                and len(_chat_cells(line)) == len(_chat_cells(lines[i + 1]))
                and len(_chat_cells(line)) >= 2):
            rows = [_chat_cells(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells = _chat_cells(lines[i])
                if len(cells) != len(rows[0]):
                    break
                rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        heading = re.match(r"^\s{0,3}(#{1,3})\s+(.+?)\s*#*\s*$", line)
        if heading:
            blocks.append((f"h{len(heading.group(1))}", heading.group(2)))
            i += 1
            continue
        if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", line):
            blocks.append(("rule", ""))
            i += 1
            continue
        item = _CHAT_LIST.match(line)
        if item:
            items: list[tuple[int, str, str]] = []
            while i < len(lines):
                match = _CHAT_LIST.match(lines[i])
                if not match:
                    break
                items.append((min(len(match.group(1)) // 2, 3), match.group(2), match.group(3)))
                i += 1
            blocks.append(("list", items))
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            next_line = lines[i]
            if (re.match(r"^\s{0,3}#{1,3}\s+", next_line)
                    or _CHAT_LIST.match(next_line)
                    or next_line.lstrip().startswith("```")
                    or ("|" in next_line and i + 1 < len(lines)
                        and all(_CHAT_TABLE_RULE.fullmatch(cell) for cell in _chat_cells(lines[i + 1])))):
                break
            paragraph.append(next_line)
            i += 1
        blocks.append(("paragraph", "\n".join(paragraph)))
    return blocks


def _chat_inline_parts(source: str) -> list[tuple[str, str]]:
    """Return visible text and Tk style tags for inline Markdown."""
    result: list[tuple[str, str]] = []
    previous = 0
    for match in _CHAT_INLINE.finditer(source):
        if match.start() > previous:
            result.append((source[previous:match.start()], ""))
        token = match.group()
        if token.startswith("`"):
            result.append((token[1:-1], "code_inline"))
        elif token.startswith(("***", "___")):
            result.append((token[3:-3], "bold_italic"))
        elif token.startswith(("**", "__")):
            result.append((token[2:-2], "bold"))
        else:
            result.append((token[1:-1], "italic"))
        previous = match.end()
    if previous < len(source):
        result.append((source[previous:], ""))
    return result


def _clean_heading(line: str, names: tuple[str, ...]) -> str:
    cleaned = line.strip().replace("**", "")
    lower = cleaned.lower()
    for name in names:
        if lower.startswith(name):
            return cleaned[len(name):].strip(" :\t")
    return cleaned


def parse_quiz_content(raw_text: str) -> dict[str, Any]:
    """Phân tích kết quả Gemini, đồng thời hỗ trợ dữ liệu cũ không hoàn toàn đúng mẫu."""
    lines = raw_text.splitlines()
    question_parts: list[str] = []
    explanation_parts: list[str] = []
    transcript_parts: list[str] = []
    options: dict[str, str] = {}
    correct = ""
    section = "question"

    for original in lines:
        line = original.strip()
        if not line:
            continue
        plain = line.replace("**", "").strip()
        lower = plain.lower()

        if lower.startswith(("transcript:", "nội dung nghe được:")):
            section = "transcript"
            value = _clean_heading(line, ("transcript", "nội dung nghe được"))
            if value:
                transcript_parts.append(value)
            continue
        if lower.startswith(("câu hỏi:", "câu:")):
            section = "question"
            value = _clean_heading(line, ("câu hỏi", "câu"))
            if value:
                question_parts.append(value)
            continue
        if lower.startswith(("lựa chọn:", "các lựa chọn:")):
            section = "options"
            continue
        if lower.startswith(("đáp án đúng:", "đáp án:")):
            section = "answer"
            value = _clean_heading(line, ("đáp án đúng", "đáp án"))
            match = re.search(r"\b([ABCD])\b", value.upper())
            if match:
                correct = match.group(1)
            continue
        if lower.startswith(("giải thích:", "lời giải:")):
            section = "explanation"
            value = _clean_heading(line, ("giải thích", "lời giải"))
            if value:
                explanation_parts.append(value)
            continue

        option_match = re.match(r"^([A-Da-d])\s*[\.\)\:\-]\s*(.+)$", plain)
        if option_match:
            letter = option_match.group(1).upper()
            options.setdefault(letter, option_match.group(2).strip())
            continue

        if section == "question" and not plain.startswith("#"):
            question_parts.append(plain)
        elif section == "answer" and not correct:
            match = re.search(r"\b([ABCD])\b", plain.upper())
            if match:
                correct = match.group(1)
        elif section == "explanation":
            explanation_parts.append(plain)
        elif section == "transcript":
            transcript_parts.append(plain)

    if not correct:
        match = re.search(
            r"(?:đáp án(?:\s+đúng)?|answer)\s*[:\-]?\s*\**\s*([ABCD])\b",
            raw_text,
            flags=re.IGNORECASE,
        )
        if match:
            correct = match.group(1).upper()

    rendered_options = []
    for letter in CHOICES:
        rendered_options.append({
            "letter": letter,
            "text": options.get(letter, f"Lựa chọn {letter} (dữ liệu cũ chưa có nội dung)"),
        })

    return {
        "transcript": " ".join(transcript_parts).strip(),
        "question": " ".join(question_parts).strip() or "Câu hỏi chưa có tiêu đề",
        "options": rendered_options,
        "correct": correct,
        "explanation": "\n".join(explanation_parts).strip()
        or "Chưa trích xuất được phần giải thích riêng. Hãy xem nội dung gốc trong lịch sử.",
    }


def _default_review_state() -> dict[str, Any]:
    return {
        "needs_review": False,
        "attempts": 0,
        "wrong_count": 0,
        "last_choice": "",
        "last_correct": None,
        "last_attempt_at": "",
        "attempt_history": [],
    }


import functools


def _norm_text(text: Any) -> str:
    return re.sub(r"[\W_]+", " ", str(text).casefold()).strip()


@functools.lru_cache(maxsize=8192)
def question_signature(answer: str) -> str:
    """Dấu vân tay của câu hỏi (bỏ qua hoa/thường, dấu câu, thứ tự đáp án). Rỗng = không đủ dữ liệu để so trùng."""
    parsed = parse_quiz_content(answer)
    question = _norm_text(parsed["question"])
    if not question or question == _norm_text("Câu hỏi chưa có tiêu đề"):
        return ""
    options = sorted(
        _norm_text(o["text"]) for o in parsed["options"]
        if "(dữ liệu cũ chưa có nội dung)" not in o["text"]
    )
    transcript = _norm_text(parsed.get("transcript", ""))
    if len(options) < 2 and not transcript:
        return ""
    return "|".join([question, transcript, "||".join(options)])


def _merge_entry_into(base: dict[str, Any], other: dict[str, Any]) -> None:
    """Gộp bản ghi trùng `other` vào `base` mà không làm mất kết quả làm bài / dấu sao."""
    br = {**_default_review_state(), **(base.get("review") or {})}
    orr = {**_default_review_state(), **(other.get("review") or {})}
    br["attempts"] = int(br.get("attempts") or 0) + int(orr.get("attempts") or 0)
    br["wrong_count"] = int(br.get("wrong_count") or 0) + int(orr.get("wrong_count") or 0)
    history = list(br.get("attempt_history") or []) + list(orr.get("attempt_history") or [])
    br["attempt_history"] = history[-100:]
    if str(orr.get("last_attempt_at") or "") > str(br.get("last_attempt_at") or ""):
        for field in ("last_choice", "last_correct", "last_attempt_at", "needs_review"):
            br[field] = orr.get(field)
    base["review"] = br
    base["is_starred"] = bool(base.get("is_starred")) or bool(other.get("is_starred"))
    base["scan_count"] = int(base.get("scan_count") or 1) + int(other.get("scan_count") or 1)
    if not base.get("screenshot") and other.get("screenshot"):
        base["screenshot"] = other["screenshot"]


def merge_duplicate_history(history: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Gộp mọi câu hỏi trùng trong lịch sử; giữ bản quét sớm nhất. Trả về (lịch sử mới, số bản ghi đã gộp)."""
    seen: dict[str, dict[str, Any]] = {}
    result: dict[str, list[dict[str, Any]]] = {}
    merged = 0
    for day in sorted(history):
        kept: list[dict[str, Any]] = []
        for item in history[day]:
            sig = question_signature(str(item.get("answer", "")))
            if sig and sig in seen:
                _merge_entry_into(seen[sig], item)
                merged += 1
                continue
            if sig:
                seen[sig] = item
            kept.append(item)
        if kept:
            result[day] = kept
    return result, merged


def normalize_history(data: Any) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """Nâng cấp lịch sử cũ theo cách cộng thêm trường, không xoá dữ liệu ban đầu."""
    if not isinstance(data, dict):
        return {}, True
    changed = False
    normalized: dict[str, list[dict[str, Any]]] = {}
    for day, items in data.items():
        if not isinstance(day, str) or not isinstance(items, list):
            changed = True
            continue
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                changed = True
                continue
            entry = dict(item)
            if not isinstance(entry.get("id"), str) or not entry.get("id"):
                entry["id"] = uuid.uuid4().hex
                changed = True
            if not isinstance(entry.get("is_starred"), bool):
                entry["is_starred"] = False
                changed = True
            review = entry.get("review")
            if not isinstance(review, dict):
                review = {}
                changed = True
            merged_review = {**_default_review_state(), **review}
            if merged_review != review:
                changed = True
            if not isinstance(merged_review.get("attempt_history"), list):
                merged_review["attempt_history"] = []
                changed = True
            entry["review"] = merged_review
            normalized_items.append(entry)
        normalized[day] = normalized_items
    return normalized, changed


def _condense_transcript(items: list[tuple[float, str]], now: float) -> str:
    """Rút gọn transcript: lời trong 2 phút gần nhất + các câu chứa từ khoá câu hỏi (≤5 phút).
    Bỏ trùng lặp, cắt còn tối đa MAX_LISTEN_CHARS ký tự (giữ phần mới nhất)."""
    picked: list[str] = []
    seen: set[str] = set()
    for t, txt in items:
        for sent in re.split(r"(?<=[.!?…])\s+", txt):
            sent = " ".join(sent.split())
            if len(sent) < 3:
                continue
            key = sent.lower()
            age = now - t
            recent = age <= LISTEN_RECENT_SEC
            keyword = age <= LISTEN_KEYWORD_SEC and any(k in key for k in QUESTION_KEYWORDS)
            if (recent or keyword) and key not in seen:
                seen.add(key)
                picked.append(sent)
    text = " ".join(picked)
    if len(text) > MAX_LISTEN_CHARS:
        text = "…" + text[-MAX_LISTEN_CHARS:].split(" ", 1)[-1]
    return text


def _review_count(history: dict[str, list[dict[str, Any]]]) -> int:
    return sum(
        1 for items in history.values() for item in items
        if item.get("review", {}).get("needs_review", False)
    )


class QuizPanel(ctk.CTkFrame):
    """Bảng làm bài dùng chung cho bài ôn theo ngày và danh sách câu sai."""

    def __init__(self, master, app: "App", review_only: bool = False) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.review_only = review_only
        self.entries: list[tuple[str, dict[str, Any]]] = []
        self.current_idx = 0
        self.answered_current = False
        self.session_answered: set[str] = set()
        self.session_correct: set[str] = set()
        self.option_buttons: dict[str, ctk.CTkButton] = {}
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)     # vùng làm bài chiếm toàn bộ chiều cao còn lại
        self._fit_job = None
        self._option_texts: dict[str, str] = {}

        toolbar = ctk.CTkFrame(self, fg_color=PALETTE["surface"], corner_radius=12)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        toolbar.grid_columnconfigure(2, weight=1)

        title = "Cần ôn tập" if self.review_only else "Chọn bộ câu hỏi"
        ctk.CTkLabel(
            toolbar, text=title, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=PALETTE["text"],
        ).grid(row=0, column=0, padx=(16, 10), pady=12)

        if not self.review_only:
            self.source_menu = ctk.CTkComboBox(
                toolbar, values=["Tất cả câu hỏi"], width=190, height=34,
                fg_color=PALETTE["surface_2"], border_color=PALETTE["border"],
                button_color=PALETTE["primary"], button_hover_color=PALETTE["primary_hover"],
                command=self.load_source,
            )
            self.source_menu.grid(row=0, column=1, padx=(0, 10), pady=10)
            source_entry = getattr(self.source_menu, "_entry", None)
            if source_entry is not None:
                source_entry.bind("<KeyPress-space>", self._on_source_space)
        else:
            self.review_mode = "wrong"
            self.review_switch = ctk.CTkSegmentedButton(
                toolbar, values=["Câu cần luyện lại (0)", "Câu hỏi đã gắn sao ⭐ (0)"],
                height=34, font=ctk.CTkFont(size=12, weight="bold"),
                selected_color=PALETTE["primary"], selected_hover_color=PALETTE["primary_hover"],
                unselected_color=PALETTE["surface_2"], unselected_hover_color=PALETTE["surface_hover"],
                command=self._on_review_mode,
            )
            self.review_switch.grid(row=0, column=1, padx=(0, 10), pady=10)
            self.review_switch.set("Câu cần luyện lại (0)")

        self.progress_label = ctk.CTkLabel(
            toolbar, text="Câu 0 / 0", text_color=PALETTE["blue"],
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.progress_label.grid(row=0, column=2, sticky="e", padx=10)
        self.score_label = ctk.CTkLabel(
            toolbar, text="Đã làm 0  •  Đúng 0", text_color=PALETTE["muted"],
            font=ctk.CTkFont(size=12),
        )
        self.score_label.grid(row=0, column=3, sticky="e", padx=(4, 16))

        self.progress = ctk.CTkProgressBar(
            self, height=6, corner_radius=3,
            progress_color=PALETTE["primary"], fg_color=PALETTE["surface_2"],
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.progress.set(0)

        self.content_scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0,
        )
        self.content_scroll.grid(row=2, column=0, sticky="nsew")
        self.content_scroll.grid_columnconfigure(0, weight=1)

        self.card = ctk.CTkFrame(
            self.content_scroll, fg_color=PALETTE["surface"], corner_radius=16,
            border_width=1, border_color=PALETTE["border"],
        )
        self.card.grid(row=0, column=0, sticky="ew")
        self.card.grid_columnconfigure(0, weight=1)

        self.question_label = ctk.CTkLabel(
            self.card, text="Chưa có câu hỏi", anchor="w", justify="left",
            wraplength=790, text_color=PALETTE["text"],
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.question_label.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 8))

        self.star_btn = ctk.CTkButton(
            self.card, text="☆", width=38, height=38, corner_radius=8,
            fg_color="transparent", hover_color=PALETTE["surface_hover"],
            border_width=1, border_color=PALETTE["border"],
            text_color=PALETTE["muted"], font=ctk.CTkFont(size=24),
            command=self.toggle_star,
        )
        self.star_btn.grid(row=0, column=1, sticky="ne", padx=(0, 18), pady=(16, 6))

        self.meta_label = ctk.CTkLabel(
            self.card, text="", anchor="w", text_color=PALETTE["muted"],
            font=ctk.CTkFont(size=11),
        )
        self.meta_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=24, pady=(0, 10))

        self.options_frame = ctk.CTkFrame(
            self.card, fg_color="transparent", corner_radius=0,
        )
        self.options_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=4)
        self.options_frame.grid_columnconfigure(0, weight=1)

        self.feedback = ctk.CTkTextbox(
            self.card, height=220, wrap="word", font=ctk.CTkFont(size=14),
            fg_color=PALETTE["surface_2"], border_width=1,
            border_color=PALETTE["border"], corner_radius=10,
            text_color=PALETTE["muted"],
        )
        self.feedback.grid(row=3, column=0, columnspan=2, sticky="ew", padx=24, pady=(12, 18))
        self._set_feedback("Chọn một đáp án để xem kết quả và lời giải thích.")

        navigation = ctk.CTkFrame(self, fg_color="transparent")
        navigation.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.prev_btn = ctk.CTkButton(
            navigation, text="←  Câu trước", width=140, height=44,
            fg_color=PALETTE["surface_2"], hover_color=PALETTE["surface_hover"],
            border_width=1, border_color=PALETTE["border"], command=self.previous,
        )
        self.prev_btn.grid(row=0, column=0)
        self.next_btn = ctk.CTkButton(
            navigation, text="Câu tiếp  →", width=140, height=44,
            fg_color=PALETTE["primary"], hover_color=PALETTE["primary_hover"],
            command=self.next,
        )
        self.next_btn.grid(row=0, column=1, padx=(10, 0))
        self.ask_btn = ctk.CTkButton(
            navigation, text="💬  Hỏi AI về câu này", width=190, height=44, cursor="hand2",
            fg_color=PALETTE["surface_2"], hover_color=PALETTE["surface_hover"],
            border_width=1, border_color=PALETTE["border"], command=self.ask_ai,
        )
        self.ask_btn.grid(row=0, column=2, padx=(10, 0))
        ctk.CTkLabel(
            navigation, text="⌨  A–D / 1–4: chọn   •   ←/→: chuyển câu",
            text_color=PALETTE["muted"], font=ctk.CTkFont(size=11),
        ).grid(row=0, column=3, padx=(16, 0))
        try:
            self.content_scroll._parent_canvas.bind("<Configure>", lambda _e: self._schedule_fit(), add="+")
        except Exception:
            pass
        self.content_scroll.bind("<Configure>", self._on_content_resize, add="+")

    def _on_content_resize(self, event) -> None:
        """Giữ nội dung xuống dòng theo chiều rộng thực tế của vùng cuộn."""
        wraplength = max(360, event.width - 130)
        self.question_label.configure(wraplength=wraplength)
        self.meta_label.configure(wraplength=wraplength)
        for letter, button in self.option_buttons.items():
            text_label = getattr(button, "_text_label", None)
            if text_label is not None:
                text_label.configure(wraplength=max(320, event.width - 150), justify="left")
            if letter in self._option_texts:
                button.configure(height=self._option_height(self._option_texts[letter], event.width))
        self._schedule_fit()

    @staticmethod
    def _option_height(text: str, width: int) -> int:
        """Ô đáp án cao thoáng; câu dài thì cao thêm theo số dòng ước tính."""
        usable = max(width, 900) - 170
        per_line = max(24, int(usable / 9.2))
        lines = max(1, -(-len(text) // per_line) + text.count("\n"))
        return max(76, 40 + 26 * lines)

    def _schedule_fit(self) -> None:
        if self._fit_job is None:
            self._fit_job = self.after(40, self._fit_feedback)

    def _fit_feedback(self) -> None:
        """Kéo giãn ô lời giải để thẻ câu hỏi lấp đầy chiều cao còn lại, sát đáy cửa sổ."""
        self._fit_job = None
        try:
            canvas_h = self.content_scroll._parent_canvas.winfo_height()
            if canvas_h < 100:
                return
            current = int(float(self.feedback.cget("height")))
            others = self.card.winfo_reqheight() - current
            target = max(180, canvas_h - others - 14)
            if abs(target - current) > 3:
                self.feedback.configure(height=target)
        except Exception:
            pass

    def _update_star_button(self, entry: dict[str, Any] | None = None) -> None:
        starred = bool(entry and entry.get("is_starred", False))
        self.star_btn.configure(
            text="★" if starred else "☆",
            text_color=PALETTE["warning"] if starred else PALETTE["muted"],
            border_color=PALETTE["warning"] if starred else PALETTE["border"],
            state="normal" if entry else "disabled",
        )

    def toggle_star(self) -> None:
        if not self.entries:
            return
        day, entry = self.entries[self.current_idx]
        self.app.toggle_star(day, entry)

    def sync_star_state(self, entry_id: str) -> None:
        if not self.entries:
            return
        _day, entry = self.entries[self.current_idx]
        if str(entry.get("id", "")) == entry_id:
            self._update_star_button(entry)

    def refresh_sources(self, preferred: str | None = None) -> None:
        if self.review_only:
            current_id = None
            if self.entries and self.current_idx < len(self.entries):
                current_id = str(self.entries[self.current_idx][1].get("id", ""))
            new_entries = (
                self.app.get_starred_entries() if self.review_mode == "starred"
                else self.app.get_review_entries()
            )
            self._update_review_counts()
            new_ids = [str(e.get("id", "")) for _d, e in new_entries]
            if current_id and current_id in new_ids and self.answered_current:
                # Giữ nguyên câu đang hiển thị (và kết quả vừa làm) nếu nó vẫn thuộc danh sách.
                self.entries = new_entries
                self.current_idx = new_ids.index(current_id)
                self.progress_label.configure(text=f"Câu {self.current_idx + 1} / {len(self.entries)}")
                self.next_btn.configure(state="normal" if self.current_idx < len(self.entries) - 1 else "disabled")
                self.prev_btn.configure(state="normal" if self.current_idx > 0 else "disabled")
                return
            self.entries = new_entries
            if current_id and current_id in new_ids:
                self.current_idx = new_ids.index(current_id)
            if self.current_idx >= len(self.entries):
                self.current_idx = max(0, len(self.entries) - 1)
            self.render()
            return

        days = sorted(self.app.history.keys(), reverse=True)
        values = ["Tất cả câu hỏi", *days]
        current = preferred or self.source_menu.get()
        self.source_menu.configure(values=values)
        if current not in values:
            current = values[0]
        self.source_menu.set(current)
        self.load_source(current)

    def _review_labels(self) -> tuple[str, str]:
        wrong = len(self.app.get_review_entries())
        starred = len(self.app.get_starred_entries())
        return f"Câu cần luyện lại ({wrong})", f"Câu hỏi đã gắn sao ⭐ ({starred})"

    def _update_review_counts(self) -> None:
        wrong_lbl, star_lbl = self._review_labels()
        self.review_switch.configure(values=[wrong_lbl, star_lbl])
        self.review_switch.set(star_lbl if self.review_mode == "starred" else wrong_lbl)

    def _on_review_mode(self, value: str) -> None:
        mode = "starred" if value.startswith("Câu hỏi đã gắn sao") else "wrong"
        if mode == self.review_mode:
            return
        self.review_mode = mode
        self.entries = []
        self.current_idx = 0
        self.session_answered.clear()
        self.session_correct.clear()
        self.answered_current = False
        self.refresh_sources()

    def load_source(self, source: str) -> None:
        self._release_input_focus()
        if source == "Tất cả câu hỏi":
            self.entries = [
                (day, item) for day in sorted(self.app.history.keys(), reverse=True)
                for item in self.app.history[day]
            ]
        else:
            self.entries = [(source, item) for item in self.app.history.get(source, [])]
        self.current_idx = 0
        self.session_answered.clear()
        self.session_correct.clear()
        self.render()

    def _set_feedback(self, text: str, tone: str = "muted") -> None:
        color = {
            "success": PALETTE["success"],
            "danger": PALETTE["danger"],
            "muted": PALETTE["muted"],
        }.get(tone, PALETTE["muted"])
        self.feedback.configure(state="normal", text_color=color)
        self.feedback.delete("1.0", "end")
        self.feedback.insert("end", text)
        self.feedback.configure(state="disabled")

    def render(self) -> None:
        try:
            self.content_scroll._parent_canvas.yview_moveto(0)
        except Exception:
            pass
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        self.option_buttons.clear()
        self._option_texts = {}
        self.answered_current = False
        self._chosen_current = ""

        total = len(self.entries)
        if total == 0:
            starred_mode = self.review_only and self.review_mode == "starred"
            if starred_mode:
                title = "Chưa có câu hỏi gắn sao"
                detail = "Bấm biểu tượng ☆ ở một câu hỏi để thêm vào danh sách này."
            elif self.review_only:
                title = "Bạn đã hoàn thành mọi câu cần ôn!"
                detail = "Các câu trả lời sai sẽ tự động xuất hiện tại đây."
            else:
                title = "Chưa có câu hỏi để luyện tập"
                detail = "Hãy quét một câu hỏi mới để bắt đầu."
            self.question_label.configure(text=title)
            self.meta_label.configure(text=detail)
            self.progress_label.configure(text="Câu 0 / 0")
            self.score_label.configure(text="Đã làm 0  •  Đúng 0")
            self.progress.set(0)
            self._set_feedback("Không có nội dung trong danh sách này.")
            self._update_star_button()
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            return

        self.current_idx = min(self.current_idx, total - 1)
        day, entry = self.entries[self.current_idx]
        parsed = parse_quiz_content(str(entry.get("answer", "")))
        self.question_label.configure(text=clean_display_text(parsed["question"]))
        self._update_star_button(entry)
        review = entry.get("review", {})
        attempts = int(review.get("attempts", 0) or 0)
        wrong = int(review.get("wrong_count", 0) or 0)
        meta = f"Ngày {day}  •  Đã làm {attempts} lần  •  Sai {wrong} lần"
        if entry.get("source") == "audio":
            meta = "🎧 Từ bài nghe  •  " + meta
        elif entry.get("source") == "screen+audio":
            meta = "📸+🎧 Ảnh + bài nghe  •  " + meta
        transcript = clean_display_text(parsed.get("transcript", "") or entry.get("listen_text", ""))
        if transcript:
            meta += f"\n🗣 Nghe được: {transcript[:300]}"
        self.meta_label.configure(text=meta, justify="left")
        self.progress_label.configure(text=f"Câu {self.current_idx + 1} / {total}")
        self.score_label.configure(
            text=f"Đã làm {len(self.session_answered)}  •  Đúng {len(self.session_correct)}"
        )
        self.progress.set((self.current_idx + 1) / total)

        for row, option in enumerate(parsed["options"]):
            letter = option["letter"]
            option_text = clean_display_text(option['text'])
            self._option_texts[letter] = option_text
            button = ctk.CTkButton(
                self.options_frame,
                text=f"{letter}      {option_text}", anchor="w",
                height=self._option_height(option_text, self.content_scroll.winfo_width()),
                font=ctk.CTkFont(size=15, weight="bold"), border_spacing=18,
                fg_color=PALETTE["surface_2"], hover_color=PALETTE["surface_hover"],
                border_width=1, border_color=PALETTE["border"], corner_radius=12,
                cursor="hand2",
                command=lambda choice=letter: self.check_answer(choice),
            )
            button.grid(row=row, column=0, sticky="ew", pady=7)
            self.option_buttons[letter] = button
        self._on_content_resize(type("ResizeEvent", (), {"width": self.content_scroll.winfo_width()})())

        if not parsed["correct"]:
            self._set_feedback(
                "Không xác định được đáp án đúng từ dữ liệu cũ. Bạn vẫn có thể xem câu hỏi trong Lịch sử đã quét.",
                "danger",
            )
            for button in self.option_buttons.values():
                button.configure(state="disabled")
        else:
            self._set_feedback("Chọn một đáp án để xem kết quả và lời giải thích.")
        self.prev_btn.configure(state="normal" if self.current_idx > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_idx < total - 1 else "disabled")

    def check_answer(self, chosen: str) -> None:
        if self.answered_current or not self.entries or chosen not in self.option_buttons:
            return
        day, entry = self.entries[self.current_idx]
        parsed = parse_quiz_content(str(entry.get("answer", "")))
        correct = parsed["correct"]
        if not correct:
            return
        self.answered_current = True
        self._chosen_current = chosen
        self._release_input_focus()
        is_correct = chosen == correct
        entry_id = str(entry.get("id", ""))
        self.session_answered.add(entry_id)
        if is_correct:
            self.session_correct.add(entry_id)

        for letter, button in self.option_buttons.items():
            button.configure(state="disabled")
            if letter == correct:
                button.configure(
                    fg_color=PALETTE["success_soft"], border_color=PALETTE["success"],
                    text_color=PALETTE["success"],
                )
            elif letter == chosen:
                button.configure(
                    fg_color=PALETTE["danger_soft"], border_color=PALETTE["danger"],
                    text_color=PALETTE["danger"],
                )

        heading = (
            f"✓ Chính xác — đáp án {correct}"
            if is_correct else f"✕ Chưa đúng — bạn chọn {chosen}, đáp án đúng là {correct}"
        )
        self._set_feedback(
            f"{heading}\n\nGiải thích của AI:\n{clean_display_text(parsed['explanation'])}",
            "success" if is_correct else "danger",
        )
        self.score_label.configure(
            text=f"Đã làm {len(self.session_answered)}  •  Đúng {len(self.session_correct)}"
        )
        self.app.record_attempt(day, entry, chosen, is_correct)

    def _release_input_focus(self) -> None:
        """Bỏ chọn ô nhập (ví dụ ô bộ câu hỏi) để phím cách luôn dùng cho 'Câu tiếp'."""
        try:
            self.winfo_toplevel().focus_set()
        except Exception:
            pass

    def _on_source_space(self, event=None):
        self.app._on_space_next(event, from_panel=True)
        return "break"

    def ask_ai(self) -> None:
        """Đưa câu hỏi hiện tại sang tab Chat với AI và nhờ AI giải thích."""
        if not self.entries:
            return
        _day, entry = self.entries[self.current_idx]
        parsed = parse_quiz_content(str(entry.get("answer", "")))
        lines = [f"Câu hỏi: {clean_display_text(parsed['question'])}", "Các lựa chọn:"]
        for option in parsed["options"]:
            if "(dữ liệu cũ chưa có nội dung)" in option["text"]:
                continue
            lines.append(f"{option['letter']}. {clean_display_text(option['text'])}")
        if parsed["correct"]:
            lines.append(f"Đáp án đúng (theo dữ liệu đã quét): {parsed['correct']}")
        chosen = getattr(self, "_chosen_current", "")
        if self.answered_current and chosen and chosen != parsed["correct"]:
            lines.append(f"Mình đã chọn: {chosen}")
        lines.append("Hãy giải thích ngắn gọn bằng tiếng Việt vì sao đáp án đó đúng và vì sao các đáp án còn lại sai.")
        title = "Hỏi: " + clean_display_text(parsed["question"])
        self.app.ask_ai_about("\n".join(lines), title)

    def handle_key(self, keysym: str) -> bool:
        """Phím tắt khi làm bài. Trả về True nếu phím đã được xử lý."""
        if not self.entries:
            return False
        if keysym == "Left":
            self.previous()
            return True
        if keysym == "Right":
            self.next()
            return True
        letters = list(self.option_buttons.keys())
        choice = ""
        if len(keysym) == 1 and keysym.upper() in letters:
            choice = keysym.upper()
        elif keysym.isdigit() and 1 <= int(keysym) <= len(letters):
            choice = letters[int(keysym) - 1]
        if not choice:
            return False
        self.check_answer(choice)
        return True

    def previous(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            self.render()

    def next(self) -> None:
        if self.current_idx < len(self.entries) - 1:
            self.current_idx += 1
            self.render()


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Study Question Tracker")
        self.geometry("1320x840")
        self.minsize(1040, 680)
        self.configure(fg_color=PALETTE["window"])
        self._setup_app_icon()

        ensure_gitignore()
        self.config_data = {**DEFAULT_CONFIG, **load_json(CONFIG_PATH, {})}
        # API key: ưu tiên kho mật khẩu; nếu config.json còn key cũ thì chuyển vào kho rồi xóa khỏi file.
        self._vault_ok = secret_available()
        self._vault_key = ""
        key_migrated = False
        file_key = str(self.config_data.get("api_key") or "")
        if self._vault_ok:
            self._vault_key = secret_get()
            if self._vault_key:
                self.config_data["api_key"] = self._vault_key
            elif file_key:
                if secret_set(file_key):
                    self._vault_key = file_key
                    key_migrated = True
                else:
                    self._vault_ok = False
        if not self.config_data.get("api_key"):
            self.config_data["api_key"] = os.environ.get("GEMINI_API_KEY", "")
        # Nâng cấp cấu hình cũ sang chuẩn mới (model đời 3, tự tắt 15s, phím tắt dịch vụ)
        saved_cfg = load_json(CONFIG_PATH, {})
        migrated_cfg = False
        normalized_model = normalize_model(self.config_data.get("model"))
        if normalized_model != self.config_data.get("model"):
            self.config_data["model"] = normalized_model
            migrated_cfg = True
        if "service_hotkey" not in saved_cfg:
            self.config_data["service_hotkey"] = DEFAULT_CONFIG["service_hotkey"]
            migrated_cfg = True
        try:
            auto_stop = float(self.config_data.get("listen_auto_stop", 15))
        except Exception:
            auto_stop = -1.0
        if auto_stop < 0 or auto_stop == 180:  # 180 là giá trị mặc định cũ
            self.config_data["listen_auto_stop"] = DEFAULT_CONFIG["listen_auto_stop"]
            migrated_cfg = True
        saved_chain = list(self.config_data.get("model_fallbacks") or [])
        if saved_chain == OLD_DEFAULT_FALLBACKS or any(m not in MODEL_CHOICES for m in saved_chain):
            # chuỗi cũ hoặc chứa model đã bị Google ngừng (vd. gemini-2.5-*) -> dùng chuỗi mặc định mới
            self.config_data["model_fallbacks"] = list(DEFAULT_FALLBACKS)
            migrated_cfg = True
        if self.config_data.get("model") not in MODEL_CHOICES:
            self.config_data["model"] = DEFAULT_MODEL
            migrated_cfg = True
        if "mouse_clicks" not in saved_cfg:   # lần đầu có tính năng: bật sẵn nhấp đúp chuột phải để quét
            self.config_data.update({"mouse_trigger": True, "mouse_button": "right", "mouse_clicks": 2})
            migrated_cfg = True
        if migrated_cfg or key_migrated or (self._vault_ok and file_key):
            self._write_config()
        loaded_history = load_json(HISTORY_PATH, {})
        self.history, migrated = normalize_history(loaded_history)
        if migrated:
            save_json(HISTORY_PATH, self.history)

        self.gemini = GeminiClient()
        self.gemini.quota = QuotaTracker(QUOTA_PATH)
        self.gemini.on_switch = self._on_model_switch
        self.listener = AudioListener(self._on_audio_segment, self._on_audio_level,
                                      on_auto_stop=self._on_listen_auto_stop)
        self._listen_busy = False
        self.hotkeys = HotkeyService(lambda: self._post(self._on_hotkey))
        self.mouse_trigger = MouseTriggerService(lambda: self._post(self._on_hotkey))
        # Phím tắt bật/tắt nghe luôn hoạt động (độc lập với nút Bật dịch vụ)
        self.listen_hotkeys = HotkeyService(lambda: self._post(self._toggle_listen))
        # Phím tắt bật/tắt dịch vụ (luôn hoạt động, kể cả khi dịch vụ đang tắt)
        self.service_hotkeys = HotkeyService(lambda: self._post(self._toggle_service))
        self.tray_icon = None
        self._quitting = False
        self.shot_vars: dict[str, ctk.BooleanVar] = {}
        self._thumbs: list[Any] = []
        self.recorder: HotkeyRecorder | None = None
        self.ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._typewriters: dict[str, dict[str, Any]] = {}
        self._chat_inflight: dict[str, Any] | None = None
        self._regen_btn: Any = None
        self.listen_log: list[tuple[float, str]] = []
        self.transcript_buffer = ""          # văn bản tạm thời, chưa gửi AI
        self._buffer_lock = threading.Lock()
        self._last_level_post = 0.0
        self._listen_consumed_until = 0.0
        # Buffer cục bộ (RAM): các đoạn audio đã ghi nhưng CHƯA gửi API lần nào
        self._audio_buffer: list[tuple[float, bytes, float]] = []
        self._stt_pending = 0
        self.local_stt = LocalTranscriber()
        self._stt_queue: "queue.Queue[tuple[float, bytes, float]]" = queue.Queue()
        threading.Thread(target=self._local_stt_worker, daemon=True).start()
        self._api_busy_until = 0.0
        self._busy = False
        self.toast = ScreenToast(self, self._bring_to_front)
        self.current_day: str | None = None

        self._build_ui()
        self._setup_global_keybindings()
        self._refresh_all()
        self._refresh_quota_label()
        safe_after(self, 60, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.bind("<Unmap>", self._on_unmap)
        self._start_listen_hotkey()
        self._start_service_hotkey()
        self._start_tray()

    def _button(self, master, text: str, command, *, kind: str = "primary", **kwargs):
        colors = {
            "primary": (PALETTE["primary"], PALETTE["primary_hover"]),
            "neutral": (PALETTE["surface_2"], PALETTE["surface_hover"]),
            "danger": (PALETTE["danger_soft"], "#512033"),
            "success": ("#166534", "#15803D"),
        }
        fg, hover = colors[kind]
        kwargs.setdefault("cursor", "hand2")
        return ctk.CTkButton(
            master, text=text, command=command, fg_color=fg, hover_color=hover,
            corner_radius=9, font=ctk.CTkFont(size=12, weight="bold"), **kwargs,
        )

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0, fg_color=PALETTE["sidebar"], height=72)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(3, weight=1)
        header.grid_propagate(False)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, padx=(20, 22), pady=12)
        if getattr(self, "_icon_pil", None) is not None:
            self._brand_image = ctk.CTkImage(light_image=self._icon_pil, dark_image=self._icon_pil, size=(40, 40))
            ctk.CTkLabel(brand, text="", image=self._brand_image, width=40, height=40).pack(side="left")
        else:
            ctk.CTkLabel(
                brand, text="ST", width=38, height=38, corner_radius=10,
                fg_color=PALETTE["primary"], text_color=PALETTE["text"],
                font=ctk.CTkFont(size=15, weight="bold"),
            ).pack(side="left")
        ctk.CTkLabel(
            brand, text="Study Question Tracker", text_color=PALETTE["text"],
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(side="left", padx=10)

        self.status_dot = ctk.CTkLabel(
            header, text="●", text_color=PALETTE["danger"], font=ctk.CTkFont(size=15),
        )
        self.status_dot.grid(row=0, column=1, padx=(0, 4))
        self.status_lbl = ctk.CTkLabel(
            header, text="Dịch vụ đang tắt", text_color=PALETTE["muted"],
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.status_lbl.grid(row=0, column=2, sticky="w")
        self.hotkey_lbl = ctk.CTkLabel(
            header, text="", text_color=PALETTE["muted"], font=ctk.CTkFont(size=11),
        )
        self.hotkey_lbl.grid(row=0, column=3, sticky="e", padx=14)
        self.toggle_btn = self._button(
            header, "Bật dịch vụ", self._toggle_service, kind="success", width=126, height=38,
        )
        self.toggle_btn.grid(row=0, column=4, padx=6)
        self.listen_btn = self._button(
            header, "🎧  Bật nghe", self._toggle_listen, kind="neutral", width=126, height=38,
        )
        self.listen_btn.grid(row=0, column=5, padx=6)
        self._button(
            header, "⚡  Quét ngay", self._on_hotkey, width=126, height=38,
        ).grid(row=0, column=6, padx=(6, 20))

        sidebar = ctk.CTkFrame(self, width=252, corner_radius=0, fg_color=PALETTE["sidebar"])
        self.sidebar = sidebar
        sidebar.grid(row=1, column=0, sticky="nsw")
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_propagate(False)
        ctk.CTkLabel(
            sidebar, text="LỊCH SỬ QUÉT", text_color=PALETTE["muted"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(20, 4))
        self.sidebar_summary = ctk.CTkLabel(
            sidebar, text="0 câu hỏi", text_color=PALETTE["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.sidebar_summary.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 10))
        self.day_list = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self.day_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        self.quota_lbl = ctk.CTkLabel(
            sidebar, text="", anchor="w", justify="left", wraplength=220,
            text_color=PALETTE["muted"], font=ctk.CTkFont(size=10),
        )
        self.quota_lbl.grid(row=3, column=0, sticky="ew", padx=18, pady=(4, 0))
        self._button(
            sidebar, "🗑  Xóa ngày đang xem", self._delete_day,
            kind="danger", height=38,
        ).grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 16))

        workspace = ctk.CTkFrame(self, fg_color=PALETTE["window"], corner_radius=0)
        workspace.grid(row=1, column=1, sticky="nsew", padx=10, pady=(8, 4))
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(1, weight=1)
        self.review_count_label = ctk.CTkLabel(
            workspace, text="", text_color=PALETTE["warning"],
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.review_count_label.grid(row=0, column=0, sticky="e", pady=(0, 6))
        self.sidebar_toggle_btn = self._button(
            workspace, "◀  Ẩn lịch sử", self._toggle_sidebar, kind="neutral", width=128, height=30)
        self.sidebar_toggle_btn.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.tabs = ctk.CTkTabview(
            workspace, fg_color=PALETTE["window"], corner_radius=12,
            segmented_button_fg_color=PALETTE["surface"],
            segmented_button_selected_color=PALETTE["primary"],
            segmented_button_selected_hover_color=PALETTE["primary_hover"],
            segmented_button_unselected_color=PALETTE["surface"],
            segmented_button_unselected_hover_color=PALETTE["surface_hover"],
        )
        self.tabs.grid(row=1, column=0, sticky="nsew")
        try:
            self.tabs._segmented_button.configure(font=ctk.CTkFont(size=13, weight="bold"), height=36)
        except Exception:
            pass

        # Thanh nghe nổi trên vùng làm việc, không chiếm thêm hàng bố cục.
        self.live_expanded = False
        self.live_widget = ctk.CTkFrame(
            workspace, width=310, height=42, fg_color=PALETTE["surface"],
            corner_radius=8, border_width=1, border_color=PALETTE["border"],
        )
        self.live_widget.place(relx=1, rely=1, x=-12, y=-12, anchor="se")
        self.live_widget.grid_propagate(False)
        self.live_widget.grid_columnconfigure(0, weight=1)
        self.live_preview = ctk.CTkLabel(
            self.live_widget, text="🎧  Đã tắt nghe", anchor="w",
            text_color=PALETTE["muted"], font=ctk.CTkFont(size=11),
        )
        self.live_preview.grid(row=0, column=0, sticky="ew", padx=(12, 4), pady=5)
        self.live_toggle = ctk.CTkButton(
            self.live_widget, text="⌃", width=30, height=28, corner_radius=6,
            fg_color=PALETTE["surface_2"], hover_color=PALETTE["surface_hover"],
            command=self._toggle_live_widget,
        )
        self.live_toggle.grid(row=0, column=1, padx=(0, 6), pady=5)
        self.live_details = ctk.CTkFrame(self.live_widget, fg_color="transparent")
        self.live_details.grid_columnconfigure(0, weight=1)
        self.live_details.grid_rowconfigure(1, weight=1)
        self.live_state = ctk.CTkLabel(
            self.live_details, text="⏸ Đã tắt nghe", text_color=PALETTE["muted"],
            font=ctk.CTkFont(size=11), anchor="w",
        )
        self.live_state.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 4))
        self.live_box = ctk.CTkTextbox(
            self.live_details, wrap="word", font=(FONT_FAMILY, 12),
            fg_color=PALETTE["window"],
        )
        self.live_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._live_box_text = ""
        self.live_box.configure(state="disabled")
        for name in ("Lịch sử đã quét", "Ôn tập & Làm bài", "Cần ôn tập", "Nhật ký nghe", "Quản lý ảnh", "Chat với AI", "Cài đặt"):
            self.tabs.add(name)

        self._build_tab_results(self.tabs.tab("Lịch sử đã quét"))
        self.quiz_panel = QuizPanel(self.tabs.tab("Ôn tập & Làm bài"), self)
        self.quiz_panel.pack(fill="both", expand=True, padx=2, pady=(4, 0))
        self.review_panel = QuizPanel(self.tabs.tab("Cần ôn tập"), self, review_only=True)
        self.review_panel.pack(fill="both", expand=True, padx=2, pady=(4, 0))
        self._build_tab_shots(self.tabs.tab("Quản lý ảnh"))
        self._build_tab_chat(self.tabs.tab("Chat với AI"))
        self._build_tab_listen_log(self.tabs.tab("Nhật ký nghe"))
        self._build_tab_settings(self.tabs.tab("Cài đặt"))
        self._update_hotkey_label()
        self._apply_sidebar_visibility(bool(self.config_data.get("sidebar_hidden", False)))

    def _apply_sidebar_visibility(self, hidden: bool) -> None:
        if hidden:
            self.sidebar.grid_remove()
        else:
            self.sidebar.grid()
        self.sidebar_toggle_btn.configure(text="☰  Hiện lịch sử" if hidden else "◀  Ẩn lịch sử")

    def _toggle_sidebar(self) -> None:
        hidden = not bool(self.config_data.get("sidebar_hidden", False))
        self.config_data["sidebar_hidden"] = hidden
        self._apply_sidebar_visibility(hidden)
        self._persist_config()

    def _toggle_live_widget(self) -> None:
        self.live_expanded = not self.live_expanded
        if self.live_expanded:
            self.live_widget.configure(height=210)
            self.live_details.grid(row=1, column=0, columnspan=2, sticky="nsew")
            self.live_widget.grid_rowconfigure(1, weight=1)
            self.live_toggle.configure(text="⌄")
        else:
            self.live_details.grid_remove()
            self.live_widget.grid_rowconfigure(1, weight=0)
            self.live_widget.configure(height=42)
            self.live_toggle.configure(text="⌃")

    def _build_tab_results(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        self.day_title = ctk.CTkLabel(
            parent, text="Chọn một ngày ở bên trái", anchor="w",
            text_color=PALETTE["text"], font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.day_title.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 12))
        self._button(parent, "🕒  Xóa theo khung giờ", self._delete_items_by_time,
                     kind="danger", width=170, height=32).grid(row=0, column=1, sticky="e", padx=8, pady=(8, 12))
        self.results_box = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.results_box.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.results_box.grid_columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ #
    # Quản lý ảnh
    # ------------------------------------------------------------------ #
    def _build_tab_shots(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)
        bar = ctk.CTkFrame(parent, fg_color=PALETTE["surface"], corner_radius=12,
                           border_width=1, border_color=PALETTE["border"])
        bar.grid(row=0, column=0, sticky="ew", padx=4, pady=(8, 6))
        self.shots_info = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=13, weight="bold"))
        self.shots_info.pack(side="left", padx=14, pady=10)
        self._button(bar, "🗑  Xóa ảnh đã chọn", self._delete_selected_shots, kind="danger", width=150, height=34).pack(side="right", padx=(6, 12))
        self._button(bar, "Bỏ chọn", lambda: self._select_all_shots(False), kind="neutral", width=80, height=34).pack(side="right", padx=4)
        self._button(bar, "☑  Chọn tất cả", lambda: self._select_all_shots(True), kind="neutral", width=110, height=34).pack(side="right", padx=4)
        self._button(bar, "⟳  Tải lại", self._refresh_shots, kind="neutral", width=90, height=34).pack(side="right", padx=4)

        rng = ctk.CTkFrame(parent, fg_color=PALETTE["surface"], corner_radius=12,
                           border_width=1, border_color=PALETTE["border"])
        rng.grid(row=1, column=0, sticky="ew", padx=4, pady=6)
        ctk.CTkLabel(rng, text="Xóa theo khung giờ  •  Ngày", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(14, 6), pady=10)
        self.range_day = ctk.CTkEntry(rng, width=110, height=32, border_color=PALETTE["border"])
        self.range_day.insert(0, _dt.date.today().isoformat())
        self.range_day.pack(side="left")
        ctk.CTkLabel(rng, text="Từ").pack(side="left", padx=(12, 4))
        self.range_from = ctk.CTkEntry(rng, width=70, height=32, border_color=PALETTE["border"])
        self.range_from.insert(0, "00:00")
        self.range_from.pack(side="left")
        ctk.CTkLabel(rng, text="đến").pack(side="left", padx=4)
        self.range_to = ctk.CTkEntry(rng, width=70, height=32, border_color=PALETTE["border"])
        self.range_to.insert(0, "23:59")
        self.range_to.pack(side="left")
        self._button(rng, "Xem trước", self._preview_range, kind="neutral", width=90, height=32).pack(side="left", padx=(12, 4))
        self._button(rng, "🗑  Xóa trong khung giờ", self._delete_range, kind="danger", width=170, height=32).pack(side="left", padx=4)
        self.range_msg = ctk.CTkLabel(rng, text="", text_color=PALETTE["muted"])
        self.range_msg.pack(side="left", padx=10)

        self.shots_grid = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.shots_grid.grid(row=2, column=0, sticky="nsew", padx=2, pady=(4, 0))
        for c in range(4):
            self.shots_grid.grid_columnconfigure(c, weight=1)
        self.tabs.configure(command=self._on_tab_changed)

    def _on_tab_changed(self) -> None:
        if self.tabs.get() == "Quản lý ảnh":
            self._refresh_shots()

    def _refresh_shots(self) -> None:
        for w in self.shots_grid.winfo_children():
            w.destroy()
        self._thumbs.clear()
        old = {k for k, v in self.shot_vars.items() if v.get()}
        self.shot_vars = {}
        shots = list_screenshots()
        self.shots_info.configure(text=f"{len(shots)} ảnh trong thư mục screenshots/")
        if not shots:
            ctk.CTkLabel(self.shots_grid, text="Chưa có ảnh nào được lưu.", text_color=PALETTE["muted"]).grid(row=0, column=0, columnspan=4, pady=30)
            return
        for i, (path, when) in enumerate(shots):
            card = ctk.CTkFrame(self.shots_grid, fg_color=PALETTE["surface"], corner_radius=12,
                                border_width=1, border_color=PALETTE["border"])
            card.grid(row=i // 4, column=i % 4, sticky="nsew", padx=6, pady=6)
            try:
                with Image.open(path) as im:
                    im.thumbnail((220, 140))
                    thumb = ctk.CTkImage(light_image=im.copy(), dark_image=im.copy(), size=im.size)
                self._thumbs.append(thumb)
                pic = ctk.CTkLabel(card, text="", image=thumb, cursor="hand2")
            except Exception:
                pic = ctk.CTkLabel(card, text="(không đọc được ảnh)", height=120, text_color=PALETTE["muted"])
            pic.pack(padx=10, pady=(10, 4))
            pic.bind("<Button-1>", lambda _e, p=path: self._open_shot(p))
            var = ctk.BooleanVar(value=str(path) in old)
            self.shot_vars[str(path)] = var
            ctk.CTkCheckBox(card, text=when.strftime("%d/%m/%Y  %H:%M:%S"), variable=var,
                            fg_color=PALETTE["primary"], hover_color=PALETTE["primary_hover"],
                            font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10, pady=(0, 10))

    def _open_shot(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except Exception as exc:
            self.toast.show("Không mở được ảnh", str(exc), PALETTE["danger"], 4000)

    def _select_all_shots(self, value: bool) -> None:
        for var in self.shot_vars.values():
            var.set(value)

    def _confirm(self, text: str) -> bool:
        from tkinter import messagebox
        return messagebox.askyesno("Xác nhận xóa", text, parent=self)

    def _remove_files(self, paths: list[Path]) -> int:
        removed = 0
        gone: set[str] = set()
        for p in paths:
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except Exception as exc:
                print("Delete error:", exc)
                continue
            gone.add(str(p.resolve()))
        if gone:
            changed = False
            for items in self.history.values():
                for item in items:
                    shot = item.get("screenshot")
                    if shot and str(Path(shot).resolve()) in gone:
                        item["screenshot"] = ""
                        changed = True
            if changed:
                save_json(HISTORY_PATH, self.history)
        self._refresh_shots()
        return removed

    def _delete_selected_shots(self) -> None:
        paths = [Path(k) for k, v in self.shot_vars.items() if v.get()]
        if not paths:
            self.toast.show("Chưa chọn ảnh nào", "", PALETTE["muted"], 2000)
            return
        if not self._confirm(f"Xóa vĩnh viễn {len(paths)} ảnh khỏi ổ cứng?"):
            return
        n = self._remove_files(paths)
        self.toast.show(f"Đã xóa {n} ảnh", "", PALETTE["success"], 2500)

    def _range_matches(self):
        try:
            return shots_in_range(list_screenshots(), self.range_day.get(), self.range_from.get(), self.range_to.get())
        except Exception as exc:
            self.range_msg.configure(text=f"Sai định dạng (YYYY-MM-DD, HH:MM): {exc}", text_color=PALETTE["danger"])
            return None

    def _preview_range(self) -> None:
        matches = self._range_matches()
        if matches is None:
            return
        wanted = {str(p) for p, _ in matches}
        for k, v in self.shot_vars.items():
            v.set(k in wanted)
        self.range_msg.configure(text=f"Có {len(matches)} ảnh trong khung giờ (đã tích chọn)", text_color=PALETTE["muted"])

    def _delete_range(self) -> None:
        matches = self._range_matches()
        if matches is None:
            return
        if not matches:
            self.range_msg.configure(text="Không có ảnh nào trong khung giờ này", text_color=PALETTE["muted"])
            return
        label = f"{self.range_day.get()} từ {self.range_from.get()} đến {self.range_to.get()}"
        if not self._confirm(f"Xóa vĩnh viễn {len(matches)} ảnh chụp ngày {label}?"):
            return
        n = self._remove_files([p for p, _ in matches])
        self.range_msg.configure(text=f"Đã xóa {n} ảnh", text_color=PALETTE["success"])

    def _bind_scroll_wheel(self, scroll: Any) -> None:
        """Giữ lại cho tương thích: cuộn chuột giờ do SmoothScroller xử lý cho mọi khung."""
        return

    def _build_tab_listen_log(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        bar.grid_columnconfigure(0, weight=1)
        self.listen_log_info = ctk.CTkLabel(
            bar, anchor="w", justify="left", text_color=PALETTE["muted"],
            text="Khi bật nghe, lời nói được chép thành chữ tại đây (chưa hỏi AI). "
                 "Bấm phím tắt quét: ảnh câu hỏi + đoạn nghe gần nhất sẽ được gửi cùng nhau.")
        self.listen_log_info.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(bar, text="Xoá nhật ký", width=110, fg_color=PALETTE["surface_2"],
                      hover_color=PALETTE["surface_hover"],
                      command=self._clear_listen_log).grid(row=0, column=1, padx=(8, 0))
        self.listen_log_box = ctk.CTkTextbox(parent, wrap="word", font=(FONT_FAMILY, 13))
        self.listen_log_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.listen_log_box.configure(state="disabled")

    # ----- Nhật ký nghe (Speech-to-Text, tách biệt khỏi việc trả lời) ----- #
    def _on_audio_level(self, level: float, speaking: bool) -> None:
        """Gọi từ luồng thu âm; CHỈ đẩy lên UI khi trạng thái đổi (tránh giựt khung)."""
        state_key = (level < 0, speaking)
        if level >= 0 and state_key == getattr(self, "_last_live_state", None):
            return
        self._last_live_state = state_key
        self._post(lambda: self._show_level(level, speaking))

    def _show_level(self, level: float, speaking: bool) -> None:
        if not hasattr(self, "live_state"):
            return
        if level < 0:
            err = getattr(self.listener, "last_error", "") or "Không mở được thiết bị âm thanh."
            self.listener.active = False
            self._set_listen_ui(False)
            self.live_state.configure(text="⚠ Lỗi thiết bị", text_color=PALETTE["danger"])
            self.live_preview.configure(text="⚠  Lỗi thiết bị")
            self.toast.show("Không mở được micro/thiết bị", err[:300], PALETTE["danger"], 8000)
            return
        self.live_state.configure(
            text="🔴 Đang ghi âm…" if speaking else "🎧 Đang nghe (lưu tạm, chưa gửi API)",
            text_color=PALETTE["danger"] if speaking else PALETTE["muted"])
        if not self.transcript_buffer:
            self.live_preview.configure(text="🔴  Đang ghi âm…" if speaking else "🎧  Đang nghe…")

    def _refresh_live_box(self) -> None:
        box = getattr(self, "live_box", None)
        if box is None:
            return
        with self._buffer_lock:
            text = self.transcript_buffer
            clips = len(self._audio_buffer)
            seconds = sum(d for _t, _b, d in self._audio_buffer)
        parts = []
        if text:
            parts.append(f"📝 Nội dung đã nghe được (cộng dồn, {len(text)} ký tự):\n{text}")
        if self._stt_pending:
            parts.append(f"⏳ Đang chép lời {self._stt_pending} đoạn…")
        if clips:
            parts.append(f"🎧 {clips} đoạn audio (~{seconds:.0f} giây) chưa chép được chữ "
                         "(cài faster-whisper để xem chữ) — sẽ gửi kèm khi quét.")
        new_text = "\n\n".join(parts) or "(Chưa có nội dung — bật 🎧 nghe để bắt đầu ghi âm tạm)"
        # Chỉ một dòng trong thanh mini; mở rộng để đọc toàn bộ lời nghe.
        latest = text.splitlines()[-1].strip() if text else ""
        preview = latest or ("Đang chép lời…" if self._stt_pending else
                             f"{clips} đoạn audio đang chờ" if clips else
                             "Đang nghe…" if self.listener_active() else "Đã tắt nghe")
        if len(preview) > 31:
            preview = preview[:30].rstrip() + "…"
        self.live_preview.configure(text=f"🎧  {preview}")
        if new_text == getattr(self, "_live_box_text", None):
            return  # không đổi gì → không vẽ lại, tránh nháy
        at_bottom = box.yview()[1] >= 0.98
        self._live_box_text = new_text
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", new_text)
        if at_bottom:
            box.see("end")
        box.configure(state="disabled")

    def _append_listen_log(self, stamp: float, text: str) -> None:
        self.listen_log.append((stamp, text))
        if len(self.listen_log) > 500:
            self.listen_log = self.listen_log[-500:]
        line = f"[{_dt.datetime.fromtimestamp(stamp).strftime('%H:%M:%S')}] {text}\n"
        box = getattr(self, "listen_log_box", None)
        if box is not None:
            box.configure(state="normal")
            box.insert("end", line)
            box.see("end")
            box.configure(state="disabled")
        self._update_listen_info()

    def _clear_listen_log(self) -> None:
        self.listen_log = []
        self._clear_audio_buffer()
        self._listen_consumed_until = time.time()
        box = getattr(self, "listen_log_box", None)
        if box is not None:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.configure(state="disabled")
        self._update_listen_info()

    def _mark_listen_log_used(self, upto: float, sent_text: str = "", clips: int = 0) -> None:
        """Chỉ gọi khi API trả kết quả THÀNH CÔNG: xoá sạch buffer đã gửi."""
        self._listen_consumed_until = max(self._listen_consumed_until, upto)
        with self._buffer_lock:
            self._audio_buffer = [it for it in self._audio_buffer if it[0] > upto]
        box = getattr(self, "listen_log_box", None)
        if box is not None:
            box.configure(state="normal")
            stamp = _dt.datetime.now().strftime('%H:%M:%S')
            body = sent_text.strip() or "(không có chữ — chỉ gửi audio thô)"
            box.insert("end", f"\n──── ✓ Đã gửi thành công lúc {stamp} ({len(sent_text)} ký tự"
                              + (f", {clips} đoạn audio" if clips else "") + ") — nội dung đã gửi: ────\n"
                              f"{body}\n──── buffer đã được xoá, bắt đầu gom đoạn mới ────\n\n")
            box.see("end")
            box.configure(state="disabled")
        self._update_listen_info()

    def _recent_listen_text(self) -> tuple[str, float]:
        """Đoạn nghe được gần đây nhất: sau lần quét trước và trong cửa sổ thời gian cấu hình."""
        try:
            window = float(self.config_data.get("listen_context_sec", 180))
        except Exception:
            window = 180.0
        now = time.time()
        since = max(self._listen_consumed_until, now - max(window, LISTEN_KEYWORD_SEC))
        items = [(t, txt.strip()) for t, txt in self.listen_log if t > since and txt.strip()]
        if not items:
            return "", now
        return _condense_transcript(items, now), items[-1][0]

    def _update_listen_info(self) -> None:
        pending, _ = self._recent_listen_text()
        with self._buffer_lock:
            self.transcript_buffer = pending
        self._refresh_live_box()
        if hasattr(self, "live_state") and not self.listener_active():
            self.live_state.configure(text="⏸ Đã tắt nghe", text_color=PALETTE["muted"])
            self.live_level.set(0)
        lbl = getattr(self, "listen_log_info", None)
        if lbl is None:
            return
        pending, _ = self._recent_listen_text()
        n = len(pending.split()) if pending else 0
        with self._buffer_lock:
            clips = len(self._audio_buffer)
            seconds = sum(d for _t, _b, d in self._audio_buffer)
        state = "đang nghe" if self.listener_active() else "đã tắt nghe"
        lbl.configure(text=f"🎧 {state}  •  Buffer chờ gửi kèm lần quét tới: "
                           f"{clips} đoạn audio (~{seconds:.0f}s)"
                           + (f", {n} từ đã chép" if n else "")
                           + "  •  không gọi API khi đang nghe")

    def _build_tab_chat(self, parent) -> None:
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        store = load_json(CHAT_HISTORY_PATH, {})
        self._chat_store = store if isinstance(store, dict) else {}
        convs = self._chat_store.get("conversations")
        self._chat_store["conversations"] = convs if isinstance(convs, list) else []
        ids = {c.get("id") for c in self._chat_store["conversations"] if isinstance(c, dict)}
        self._chat_active_id = self._chat_store.get("active") if self._chat_store.get("active") in ids else None
        if self._chat_active_id is None and self._chat_store["conversations"]:
            self._chat_active_id = self._chat_sorted()[0]["id"]

        side = ctk.CTkFrame(parent, width=240, fg_color=PALETTE["sidebar"], corner_radius=12)
        self.chat_side = side
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(8, 0))
        side.grid_propagate(False)
        side.grid_rowconfigure(1, weight=1)
        side.grid_columnconfigure(0, weight=1)
        self._button(side, "＋  Chat mới", self._new_chat, height=38).grid(
            row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        self.chat_list = ctk.CTkScrollableFrame(side, fg_color="transparent")
        self.chat_list.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 6))
        self.chat_list.grid_columnconfigure(0, weight=1)
        self._button(side, "🗑  Xóa tất cả lịch sử", self._delete_all_chats, kind="neutral", height=32).grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        main = ctk.CTkFrame(parent, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # Thanh đầu: nút ẩn/hiện danh sách luôn nằm ở góc trên bên trái, không bị che.
        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.chat_list_toggle = self._button(
            head, "◀  Ẩn danh sách", self._toggle_chat_list, kind="neutral", width=140, height=30)
        self.chat_list_toggle.pack(side="left")
        self.chat_title_label = ctk.CTkLabel(head, text="", anchor="w", text_color=PALETTE["muted"])
        self.chat_title_label.pack(side="left", padx=12)

        self.chat_view = ctk.CTkTextbox(
            main, wrap="word", font=ctk.CTkFont(size=13),
            fg_color=PALETTE["surface"], border_width=1,
            border_color=PALETTE["border"], corner_radius=12,
        )
        self.chat_view.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 12))
        self._configure_chat_text()
        self.chat_entry = ctk.CTkEntry(
            main, placeholder_text="Nhập nội dung cần trao đổi…", height=42,
            fg_color=PALETTE["surface"], border_color=PALETTE["border"],
        )
        # pady dưới chừa chỗ cho thanh "Nghe" nổi ở góc dưới bên phải, để không che nút Gửi.
        self.chat_entry.grid(row=2, column=0, sticky="ew", padx=(0, 8), pady=(0, 56))
        self.chat_entry.bind("<Return>", self._send_chat_from_entry)
        self.chat_entry.bind("<FocusIn>", self._chat_entry_focused, add="+")
        self._style_chat_entry_selection()
        self.chat_send_btn = self._button(main, "Gửi  →", self._on_send_or_stop, width=96, height=42)
        self.chat_send_btn.grid(row=2, column=1, pady=(0, 56))
        self._button(
            main, "Xóa chat này", self._clear_chat, kind="neutral", width=130, height=42,
        ).grid(row=2, column=2, padx=(8, 0), pady=(0, 56))

        self._render_chat_view()
        self._refresh_chat_list()
        self._apply_chat_list_visibility(bool(self.config_data.get("chat_list_hidden", False)))

    def _apply_chat_list_visibility(self, hidden: bool) -> None:
        if hidden:
            self.chat_side.grid_remove()
        else:
            self.chat_side.grid()
        self.chat_list_toggle.configure(text="☰  Hiện danh sách" if hidden else "◀  Ẩn danh sách")

    def _toggle_chat_list(self) -> None:
        hidden = not bool(self.config_data.get("chat_list_hidden", False))
        self.config_data["chat_list_hidden"] = hidden
        self._apply_chat_list_visibility(hidden)
        self._persist_config()

    def _build_tab_settings(self, parent) -> None:
        # Toàn bộ tuỳ chọn nằm trong khung cuộn dọc -> xem được ở mọi kích thước cửa sổ
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(
            parent, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=PALETTE["border"],
            scrollbar_button_hover_color=PALETTE["surface_hover"],
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        self.settings_scroll = scroll
        parent = scroll
        card = ctk.CTkFrame(
            parent, fg_color=PALETTE["surface"], corner_radius=14,
            border_width=1, border_color=PALETTE["border"],
        )
        card.grid(row=0, column=0, sticky="new", padx=(6, 10), pady=10)
        card.grid_columnconfigure(1, weight=1)
        pad = {"padx": 16, "pady": 10}
        ctk.CTkLabel(card, text="Gemini API Key", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", **pad)
        api_box = ctk.CTkFrame(card, fg_color="transparent")
        api_box.grid(row=0, column=1, sticky="ew", **pad)
        self.api_entry = ctk.CTkEntry(api_box, show="•", width=440, height=38, border_color=PALETTE["border"])
        self.api_entry.insert(0, self.config_data["api_key"])
        self.api_entry.pack(fill="x")
        status, status_color = self._api_status_text()
        self.api_status = ctk.CTkLabel(api_box, text=status, anchor="w", justify="left", wraplength=560,
                                       text_color=status_color, font=ctk.CTkFont(size=11))
        self.api_status.pack(anchor="w", pady=(4, 0))
        ctk.CTkLabel(card, text="Model AI", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, sticky="w", **pad)
        self.model_menu = ctk.CTkComboBox(
            card, width=280, height=38,
            values=MODEL_CHOICES,
            border_color=PALETTE["border"], button_color=PALETTE["primary"],
        )
        self.model_menu.set(self.config_data["model"])
        self.model_menu.grid(row=1, column=1, sticky="w", **pad)
        self.speed_menu = ctk.CTkSegmentedButton(
            card, values=["⚡ Nhanh", "🎯 Chính xác"], height=34,
            selected_color=PALETTE["primary"], selected_hover_color=PALETTE["primary_hover"],
            unselected_color=PALETTE["surface_2"], unselected_hover_color=PALETTE["surface_hover"],
        )
        self.speed_menu.set("🎯 Chính xác" if self.config_data.get("ai_speed") == "accurate" else "⚡ Nhanh")
        self.speed_menu.grid(row=1, column=1, sticky="e", **pad)
        ctk.CTkLabel(card, text="Phím tắt toàn cục", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", **pad)
        hotkey_row = ctk.CTkFrame(card, fg_color="transparent")
        hotkey_row.grid(row=2, column=1, sticky="ew", **pad)
        self.hotkey_entry = ctk.CTkEntry(hotkey_row, width=240, height=38, border_color=PALETTE["border"])
        self.hotkey_entry.insert(0, self.config_data["hotkey"])
        self.hotkey_entry.pack(side="left")
        self.record_btn = self._button(
            hotkey_row, "⌨  Ghi tổ hợp phím", self._record_hotkey,
            kind="neutral", width=160, height=38,
        )
        self.record_btn.pack(side="left", padx=10)
        hk_info = ctk.CTkFrame(card, fg_color="transparent")
        hk_info.grid(row=3, column=1, sticky="ew", padx=16)
        ctk.CTkLabel(hk_info, text="Ví dụ: <ctrl>+<shift>+x  •  <f2>  •  <alt>+q", text_color=PALETTE["muted"]).pack(anchor="w")
        ctk.CTkLabel(hk_info, text="⚡ Phím tắt bật/tắt dịch vụ", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 4))
        service_hk_row = ctk.CTkFrame(hk_info, fg_color="transparent")
        service_hk_row.pack(anchor="w", pady=(0, 6))
        self.service_hotkey_entry = ctk.CTkEntry(service_hk_row, width=240, height=38, border_color=PALETTE["border"])
        self.service_hotkey_entry.insert(0, self.config_data.get("service_hotkey", ""))
        self.service_hotkey_entry.pack(side="left")
        self.service_record_btn = self._button(
            service_hk_row, "⌨  Ghi tổ hợp phím", self._record_service_hotkey,
            kind="neutral", width=160, height=38,
        )
        self.service_record_btn.pack(side="left", padx=10)
        ctk.CTkLabel(hk_info, text="Mặc định <ctrl>+<shift>+k — bật/tắt nhanh dịch vụ. Để trống để tắt.",
                     text_color=PALETTE["muted"]).pack(anchor="w")
        ctk.CTkLabel(hk_info, text="🎧 Phím tắt bật/tắt nghe", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 4))
        listen_hk_row = ctk.CTkFrame(hk_info, fg_color="transparent")
        listen_hk_row.pack(anchor="w", pady=(0, 6))
        self.listen_hotkey_entry = ctk.CTkEntry(listen_hk_row, width=240, height=38, border_color=PALETTE["border"])
        self.listen_hotkey_entry.insert(0, self.config_data.get("listen_hotkey", ""))
        self.listen_hotkey_entry.pack(side="left")
        self.listen_record_btn = self._button(
            listen_hk_row, "⌨  Ghi tổ hợp phím", self._record_listen_hotkey,
            kind="neutral", width=160, height=38,
        )
        self.listen_record_btn.pack(side="left", padx=10)
        ctk.CTkLabel(hk_info, text="Luôn hoạt động kể cả khi ứng dụng đang ẩn. Để trống để tắt.", text_color=PALETTE["muted"]).pack(anchor="w")
        self.tray_var = ctk.BooleanVar(value=bool(self.config_data.get("minimize_to_tray", True)))
        ctk.CTkCheckBox(
            hk_info, text="Thu nhỏ cửa sổ (nút –) thì chạy ngầm ở khay hệ thống; nút X luôn thoát hẳn",
            variable=self.tray_var, fg_color=PALETTE["primary"],
        ).pack(anchor="w", pady=(10, 4))
        self._button(hk_info, "⬇  Ẩn xuống khay ngay", self._hide_to_tray,
                     kind="neutral", width=180, height=34).pack(anchor="w", pady=(4, 6))
        ctk.CTkLabel(card, text="Chế độ quét", font=ctk.CTkFont(weight="bold")).grid(row=4, column=0, sticky="nw", **pad)
        mode_row = ctk.CTkFrame(card, fg_color="transparent")
        mode_row.grid(row=4, column=1, sticky="ew", **pad)
        self.mode_var = ctk.StringVar(value=self.config_data.get("scan_mode", "select"))
        for value, label in (("select", "Kéo chọn vùng"), ("fixed", "Vùng cố định"), ("full", "Toàn màn hình")):
            ctk.CTkRadioButton(
                mode_row, text=label, value=value, variable=self.mode_var,
                command=self._mode_changed, fg_color=PALETTE["primary"], hover_color=PALETTE["primary_hover"],
            ).pack(side="left", padx=(0, 18))
        region_row = ctk.CTkFrame(card, fg_color="transparent")
        region_row.grid(row=5, column=1, sticky="ew", padx=16)
        self.region_lbl = ctk.CTkLabel(region_row, text=self._region_text(), text_color=PALETTE["muted"])
        self.region_lbl.pack(side="left")
        self._button(
            region_row, "▣  Đặt vùng cố định", self._pick_region,
            kind="neutral", width=160, height=36,
        ).pack(side="left", padx=12)
        self.save_shots_var = ctk.BooleanVar(value=bool(self.config_data["save_screenshots"]))
        ctk.CTkCheckBox(
            card, text="Lưu ảnh chụp vào thư mục screenshots/",
            variable=self.save_shots_var, fg_color=PALETTE["primary"], hover_color=PALETTE["primary_hover"],
        ).grid(row=6, column=1, sticky="w", **pad)
        ctk.CTkLabel(card, text="Chụp bằng chuột", font=ctk.CTkFont(weight="bold")).grid(row=7, column=0, sticky="w", **pad)
        mouse_row = ctk.CTkFrame(card, fg_color="transparent")
        mouse_row.grid(row=7, column=1, sticky="ew", **pad)
        self.mouse_var = ctk.BooleanVar(value=bool(self.config_data.get("mouse_trigger")))
        ctk.CTkCheckBox(mouse_row, text="Bật chụp khi nhấn", variable=self.mouse_var,
                        fg_color=PALETTE["primary"], hover_color=PALETTE["primary_hover"]).pack(side="left")
        self.mouse_menu = ctk.CTkComboBox(mouse_row, width=150, height=34, state="readonly",
                                          values=list(MOUSE_BUTTONS.values()),
                                          border_color=PALETTE["border"], button_color=PALETTE["primary"])
        self.mouse_menu.set(MOUSE_BUTTONS.get(self.config_data.get("mouse_button", "right"), "Chuột phải"))
        self.mouse_menu.pack(side="left", padx=10)
        self.mouse_clicks_menu = ctk.CTkComboBox(mouse_row, width=170, height=34, state="readonly",
                                                 values=list(MOUSE_CLICKS.values()),
                                                 border_color=PALETTE["border"], button_color=PALETTE["primary"])
        self.mouse_clicks_menu.set(MOUSE_CLICKS.get(int(self.config_data.get("mouse_clicks", 2) or 2), MOUSE_CLICKS[2]))
        self.mouse_clicks_menu.pack(side="left")
        ctk.CTkLabel(card, text="Lưu ý: chỉ hoạt động khi đã bấm “Bật dịch vụ”. Nhấp đúp = 2 lần liên tiếp nhanh, ở bất kỳ đâu trên màn hình (cách nhau ≥ 1,5 giây).",
                     text_color=PALETTE["muted"]).grid(row=8, column=1, sticky="w", padx=16)
        # ------------------------- Chế độ nghe audio ------------------------- #
        ctk.CTkLabel(card, text="Nghe audio (Listening)", font=ctk.CTkFont(weight="bold")).grid(row=9, column=0, sticky="nw", **pad)
        listen_row = ctk.CTkFrame(card, fg_color="transparent")
        listen_row.grid(row=9, column=1, sticky="ew", **pad)
        devices = audio_input_devices()
        self.listen_device_menu = ctk.CTkComboBox(
            listen_row, width=300, height=34, state="readonly",
            values=devices or ["(Không tìm thấy thiết bị thu âm)"],
            border_color=PALETTE["border"], button_color=PALETTE["primary"],
        )
        saved_device = self.config_data.get("listen_device", "")
        self.listen_device_menu.set(saved_device if saved_device in devices else (devices[0] if devices else "(Không tìm thấy thiết bị thu âm)"))
        self.listen_device_menu.pack(side="left")
        self._button(listen_row, "⟳", lambda: self._reload_audio_devices(), kind="neutral", width=40, height=34).pack(side="left", padx=6)
        self.listen_lang_menu = ctk.CTkComboBox(
            listen_row, width=150, height=34, state="readonly",
            values=["Tự nhận diện", "Tiếng Việt", "Tiếng Anh"],
            border_color=PALETTE["border"], button_color=PALETTE["primary"],
        )
        self.listen_lang_menu.set({"vi": "Tiếng Việt", "en": "Tiếng Anh"}.get(
            self.config_data.get("listen_language", "auto"), "Tự nhận diện"))
        self.listen_lang_menu.pack(side="left", padx=6)

        tune_row = ctk.CTkFrame(card, fg_color="transparent")
        tune_row.grid(row=10, column=1, sticky="ew", padx=16)
        ctk.CTkLabel(tune_row, text="Độ nhạy").pack(side="left")
        self.listen_threshold_entry = ctk.CTkEntry(tune_row, width=70, height=32, border_color=PALETTE["border"])
        self.listen_threshold_entry.insert(0, str(self.config_data.get("listen_threshold", 0.012)))
        self.listen_threshold_entry.pack(side="left", padx=(6, 14))
        ctk.CTkLabel(tune_row, text="Giây im lặng để chốt câu").pack(side="left")
        self.listen_silence_entry = ctk.CTkEntry(tune_row, width=70, height=32, border_color=PALETTE["border"])
        self.listen_silence_entry.insert(0, str(self.config_data.get("listen_silence", 1.2)))
        self.listen_silence_entry.pack(side="left", padx=6)
        self.listen_save_var = ctk.BooleanVar(value=bool(self.config_data.get("listen_save_audio")))
        ctk.CTkCheckBox(tune_row, text="Lưu file audio", variable=self.listen_save_var,
                        fg_color=PALETTE["primary"], hover_color=PALETTE["primary_hover"]).pack(side="left", padx=14)

        ctk.CTkLabel(
            card,
            text=("Mẹo: để nghe tiếng phát ra từ máy tính, chọn thiết bị loopback "
                  "(Stereo Mix / VB-Cable / What U Hear). Cần: pip install sounddevice numpy"),
            text_color=PALETTE["muted"], wraplength=600, justify="left",
        ).grid(row=11, column=1, sticky="w", padx=16)

        ctk.CTkLabel(card, text="Hạn mức AI", font=ctk.CTkFont(weight="bold")).grid(row=12, column=0, sticky="nw", **pad)
        quota_box = ctk.CTkFrame(card, fg_color="transparent")
        quota_box.grid(row=12, column=1, sticky="ew", **pad)
        self.fallback_var = ctk.BooleanVar(value=bool(self.config_data.get("auto_fallback", True)))
        ctk.CTkCheckBox(quota_box, text="Hết lượt miễn phí thì tự chuyển sang model dự phòng",
                        variable=self.fallback_var, fg_color=PALETTE["primary"],
                        hover_color=PALETTE["primary_hover"]).pack(anchor="w")
        ctk.CTkLabel(quota_box, text="Thứ tự model dự phòng (ngăn cách bằng dấu phẩy):",
                     text_color=PALETTE["muted"], font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(8, 2))
        self.fallback_entry = ctk.CTkEntry(quota_box, height=34, border_color=PALETTE["border"])
        self.fallback_entry.insert(0, ", ".join(self.config_data.get("model_fallbacks") or DEFAULT_FALLBACKS))
        self.fallback_entry.pack(fill="x")
        ctk.CTkLabel(quota_box, text="Mỗi model có hạn mức miễn phí riêng; hết lượt theo ngày sẽ được cấp lại lúc "
                                     "khoảng 14h–15h (giờ Việt Nam).",
                     text_color=PALETTE["muted"], font=ctk.CTkFont(size=11), wraplength=560,
                     justify="left").pack(anchor="w", pady=(4, 0))
        rpm_row = ctk.CTkFrame(quota_box, fg_color="transparent")
        rpm_row.pack(anchor="w", pady=(8, 0))
        ctk.CTkLabel(rpm_row, text="Giữ nhịp tối đa", text_color=PALETTE["muted"],
                     font=ctk.CTkFont(size=11)).pack(side="left")
        self.rpm_entry = ctk.CTkEntry(rpm_row, width=56, height=30, border_color=PALETTE["border"])
        self.rpm_entry.insert(0, str(self.config_data.get("rpm_limit", 5)))
        self.rpm_entry.pack(side="left", padx=6)
        ctk.CTkLabel(rpm_row, text="lượt/phút mỗi model (gói miễn phí = 5; có thanh toán thì đặt 0 để tắt)",
                     text_color=PALETTE["muted"], font=ctk.CTkFont(size=11)).pack(side="left")
        self.quota_settings_lbl = ctk.CTkLabel(quota_box, text="", anchor="w", justify="left",
                                               text_color=PALETTE["text"], font=ctk.CTkFont(size=11))
        self.quota_settings_lbl.pack(anchor="w", pady=(8, 0))
        self._button(quota_box, "↗  Xem hạn mức thật trên Google AI Studio",
                     lambda: __import__("webbrowser").open("https://aistudio.google.com/rate-limit"),
                     kind="neutral", width=300, height=32).pack(anchor="w", pady=(8, 0))
        self._refresh_quota_label()

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=15, column=1, sticky="w", **pad)
        self._button(btn_row, "✓  Lưu cấu hình", self._save_settings, width=160, height=42).pack(side="left")
        self._button(btn_row, "🧹  Gộp câu hỏi trùng", self._merge_duplicates_clicked,
                     kind="neutral", width=190, height=42).pack(side="left", padx=(10, 0))
        self.settings_msg = ctk.CTkLabel(card, text="", text_color=PALETTE["success"], font=ctk.CTkFont(weight="bold"))
        self.settings_msg.grid(row=16, column=1, sticky="w", padx=16, pady=(0, 16))

    def _region_text(self) -> str:
        region = self.config_data.get("region")
        return f"Vùng cố định: {region[2]}×{region[3]} tại ({region[0]}, {region[1]})" if region else "Chưa đặt vùng cố định"

    def _update_hotkey_label(self) -> None:
        extra = ""
        if self.config_data.get("mouse_trigger"):
            double = " ×2" if int(self.config_data.get("mouse_clicks", 1) or 1) >= 2 else ""
            extra = f"  •  {MOUSE_BUTTONS.get(self.config_data.get('mouse_button', 'right'), 'Chuột phải')}{double}"
        svc = (self.config_data.get("service_hotkey") or "").strip()
        if svc:
            extra += f"  •  Bật/tắt dịch vụ {svc}"
        self.hotkey_lbl.configure(text=f"Phím tắt  {self.config_data['hotkey']}{extra}")

    def _sync_mouse(self) -> None:
        if self.hotkeys.active and self.config_data.get("mouse_trigger"):
            self.mouse_trigger.start(self.config_data.get("mouse_button", "right"),
                                     int(self.config_data.get("mouse_clicks", 1) or 1))
        else:
            self.mouse_trigger.stop()

    def _drain_queue(self) -> None:
        while True:
            try:
                self.ui_queue.get_nowait()()
            except queue.Empty:
                break
            except Exception as exc:
                print("UI error:", exc)
        safe_after(self, 60, self._drain_queue)

    def _post(self, fn: Callable[[], None]) -> None:
        self.ui_queue.put(fn)

    def _toast(self, text: str, ok: bool = True) -> None:
        self.settings_msg.configure(text=text, text_color=PALETTE["success"] if ok else PALETTE["danger"])
        safe_after(self, 4000, lambda: self.settings_msg.configure(text=""))

    def _toggle_service(self) -> None:
        if self.hotkeys.active:
            self.hotkeys.stop()
            self.mouse_trigger.stop()
            self._set_service_ui(False)
            self.toast.show("⏹ Đã tắt dịch vụ", "", PALETTE["muted"], 2500)
            return
        try:
            self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
            self.hotkeys.start(self.config_data["hotkey"])
            self._sync_mouse()
        except Exception as exc:
            self.tabs.set("Cài đặt")
            self._toast(f"Không bật được: {exc}", ok=False)
            self.toast.show("Không bật được dịch vụ", str(exc), PALETTE["danger"], 5000)
            return
        self._set_service_ui(True)
        self.toast.show("▶ Dịch vụ đang bật",
                        f"Phím tắt quét: {self.config_data.get('hotkey', '')}",
                        PALETTE["success"], 2500)

    # ------------------------------------------------------------------ #
    # Chế độ nghe audio -> transcribe -> trả lời tự động
    # ------------------------------------------------------------------ #
    def _reload_audio_devices(self) -> None:
        devices = audio_input_devices()
        self.listen_device_menu.configure(values=devices or ["(Không tìm thấy thiết bị thu âm)"])
        if devices:
            current = self.config_data.get("listen_device", "")
            self.listen_device_menu.set(current if current in devices else devices[0])
        self._toast(f"Tìm thấy {len(devices)} thiết bị thu âm.")

    def _listen_language(self) -> str:
        return {"Tiếng Việt": "vi", "Tiếng Anh": "en"}.get(
            self.listen_lang_menu.get(), "auto") if hasattr(self, "listen_lang_menu") else "auto"

    def _toggle_listen(self) -> None:
        if self.listener.active:
            self.listener.stop()
            self._set_listen_ui(False)
            self.toast.show("🎧 Đã tắt chế độ nghe", "", PALETTE["muted"], 2500)
            return
        try:
            self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
            device = self.listen_device_menu.get() if hasattr(self, "listen_device_menu") else ""
            if device.startswith("("):
                device = ""
            try:
                threshold = float(self.listen_threshold_entry.get())
            except Exception:
                threshold = float(DEFAULT_CONFIG["listen_threshold"])
            try:
                silence = float(self.listen_silence_entry.get())
            except Exception:
                silence = float(DEFAULT_CONFIG["listen_silence"])
            self.config_data["listen_device"] = device
            self.config_data["listen_language"] = self._listen_language()
            self.config_data["listen_threshold"] = threshold
            self.config_data["listen_silence"] = silence
            self.config_data["listen_save_audio"] = bool(self.listen_save_var.get())
            self._persist_config()
            try:
                self.listener.auto_stop_sec = float(self.config_data.get("listen_auto_stop", 15))
            except Exception:
                self.listener.auto_stop_sec = float(DEFAULT_CONFIG["listen_auto_stop"])
            self.listener.start(device, threshold, silence)
        except Exception as exc:
            self.tabs.set("Cài đặt")
            self._toast(f"Không bật được chế độ nghe: {exc}", ok=False)
            return
        # Phiên nghe mới: chỉ gom dữ liệu từ lần bật gần nhất
        self._listen_consumed_until = time.time()
        self._clear_audio_buffer()
        self._append_listen_log(time.time(), "──── 🎧 Bắt đầu phiên nghe mới ────"
                                if self.local_stt.available else
                                "──── 🎧 Bắt đầu nghe (chưa cài faster-whisper: chỉ lưu audio, không hiện chữ) ────")
        self._listen_consumed_until = time.time()
        self._set_listen_ui(True)
        self.toast.show("🎧 Đang nghe câu hỏi…",
                        "Nội dung được lưu tạm trong bộ nhớ và chỉ gửi kèm khi bạn quét câu hỏi.",
                        PALETTE["primary"], 4000)

    def _on_listen_auto_stop(self) -> None:
        """Gọi từ luồng thu âm khi im lặng quá lâu: tự tắt nghe và báo lên UI."""
        def handle() -> None:
            if not self.listener.active:
                self._set_listen_ui(False)
                self.live_state.configure(text="⏸ Tự tắt (không có âm thanh)", text_color=PALETTE["muted"])
                self._append_listen_log(time.time(), "──── ⏸ Tự tắt ghi âm vì không nghe thấy âm thanh ────")
                self.toast.show("🎧 Đã tự tắt ghi âm",
                                "Không nghe thấy âm thanh nào trong một lúc lâu nên chế độ nghe đã tự tắt.",
                                PALETTE["warning"], 5000)
        self._post(handle)

    def _set_listen_ui(self, active: bool) -> None:
        self._post(self._update_listen_info)
        if active:
            self.listen_btn.configure(text="🎧  Đang nghe", fg_color=PALETTE["success"], hover_color="#15803D")
            self.status_lbl.configure(text="Đang nghe audio…")
            self.status_dot.configure(text_color=PALETTE["success"])
        else:
            self.listen_btn.configure(text="🎧  Bật nghe", fg_color=PALETTE["surface_2"],
                                      hover_color=PALETTE["surface_hover"])
            self._set_service_ui(self.hotkeys.active)

    def _on_audio_segment(self, wav_bytes: bytes, duration: float) -> None:
        """Gọi từ luồng thu âm: CHỈ lưu đoạn ghi âm vào buffer trong RAM.

        Không gọi Gemini ở đây - tránh hoàn toàn lỗi 429 do gọi API liên tục.
        Toàn bộ buffer chỉ được gửi 1 lần duy nhất khi người dùng quét câu hỏi.
        """
        stamp = time.time()
        if self.config_data.get("listen_save_audio"):
            try:
                AUDIO_DIR.mkdir(exist_ok=True)
                name = _dt.datetime.fromtimestamp(stamp).strftime("%Y%m%d_%H%M%S")
                with open(AUDIO_DIR / f"listen_{name}.wav", "wb") as fh:
                    fh.write(wav_bytes)
            except Exception as exc:
                print("Không lưu được file audio:", exc)

        if self.local_stt.available:
            self._stt_pending += 1
            self._stt_queue.put((stamp, wav_bytes, duration))
            self._post(self._update_listen_info)
            return
        with self._buffer_lock:
            self._audio_buffer.append((stamp, wav_bytes, duration))
            self._trim_audio_buffer_locked()
        self._post(lambda: self._append_listen_log(
            stamp, f"(🎧 đoạn audio {duration:.0f}s đã lưu — chưa chép được chữ vì chưa cài faster-whisper)"))

    def _local_stt_worker(self) -> None:
        """Chép lời OFFLINE từng đoạn, cộng dồn chữ vào buffer + hiện lên nhật ký."""
        while True:
            stamp, wav, duration = self._stt_queue.get()
            text = self.local_stt.transcribe(wav, self.config_data.get("listen_language", "auto"))
            self._stt_pending = max(0, self._stt_pending - 1)
            if text:
                self._post(lambda s=stamp, t=text: self._append_listen_log(s, t))
            elif not self.local_stt.available:
                with self._buffer_lock:   # lỗi model -> giữ audio thô để gửi kèm
                    self._audio_buffer.append((stamp, wav, duration))
                    self._trim_audio_buffer_locked()
                self._post(self._update_listen_info)
            else:
                self._post(self._update_listen_info)

    def _trim_audio_buffer_locked(self) -> None:
        """Giới hạn buffer theo cửa sổ thời gian và dung lượng để không phình RAM."""
        try:
            window = float(self.config_data.get("listen_context_sec", 180))
        except Exception:
            window = 180.0
        now = time.time()
        self._audio_buffer = [item for item in self._audio_buffer if now - item[0] <= window]
        total = sum(len(b) for _t, b, _d in self._audio_buffer)
        secs = sum(d for _t, _b, d in self._audio_buffer)
        while self._audio_buffer and (total > MAX_AUDIO_BYTES or secs > MAX_AUDIO_SECONDS):
            _t, b, d = self._audio_buffer.pop(0)
            total -= len(b)
            secs -= d

    def _take_listen_payload(self) -> tuple[str, list[bytes], float, float]:
        """Lấy (text buffer, các đoạn audio, mốc thời gian, tổng số giây) để gửi kèm 1 lần."""
        text, upto = self._recent_listen_text()
        with self._buffer_lock:
            self._trim_audio_buffer_locked()
            items = list(self._audio_buffer)
        # Audio thô KHÔNG được gửi lên API — chỉ dùng để chép lời cục bộ.
        if items:
            upto = max(upto, items[-1][0])
        return text, [], upto, 0.0

    def _clear_audio_buffer(self) -> None:
        with self._buffer_lock:
            self._audio_buffer = []

    def solve_text_question(self, content: str) -> None:
        """Gửi một câu hỏi dạng text (ví dụ transcript đã có) cho AI và lưu kết quả."""
        def work() -> None:
            try:
                self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
                answer = self.gemini.solve_text(content)
                entry = {
                    "id": uuid.uuid4().hex,
                    "time": _dt.datetime.now().strftime("%H:%M:%S"),
                    "answer": answer,
                    "screenshot": "",
                    "source": "text",
                    "review": _default_review_state(),
                }
                self._post(lambda: self._add_entry(entry))
            except ApiBusyError as exc:
                message = str(exc)
                self._post(lambda: self.toast.show("⏳ Đang vượt quá giới hạn API", message[:300],
                                                   PALETTE["warning"], 10000))
            except Exception as exc:
                message = friendly_api_error(exc)
                print("Lỗi gửi câu hỏi:", type(exc).__name__, message[:200])
                self._post(lambda: self.toast.show("Lỗi gửi câu hỏi", message[:300], PALETTE["danger"], 8000))

        threading.Thread(target=work, daemon=True).start()

    def _set_service_ui(self, active: bool) -> None:
        if active:
            self.toggle_btn.configure(text="Tắt dịch vụ", fg_color=PALETTE["danger"], hover_color="#E11D48")
            self.status_dot.configure(text_color=PALETTE["success"])
            self.status_lbl.configure(text="Dịch vụ đang bật")
        else:
            self.toggle_btn.configure(text="Bật dịch vụ", fg_color="#166534", hover_color="#15803D")
            self.status_dot.configure(text_color=PALETTE["danger"])
            self.status_lbl.configure(text="Dịch vụ đang tắt")

    def _record_hotkey(self) -> None:
        self.record_btn.configure(text="Đang nghe…")
        was_active = self.hotkeys.active
        self.hotkeys.stop()
        def done(combo: str) -> None:
            def apply() -> None:
                self.hotkey_entry.delete(0, "end")
                self.hotkey_entry.insert(0, combo)
                self.record_btn.configure(text="⌨  Ghi tổ hợp phím")
                if was_active:
                    self._set_service_ui(False)
                    self._toast("Đã ghi phím tắt — hãy lưu rồi bật lại.")
            self._post(apply)
        self.recorder = HotkeyRecorder(done)
        self.recorder.start()

    def _record_listen_hotkey(self) -> None:
        self.listen_record_btn.configure(text="Đang nghe…")
        self.listen_hotkeys.stop()
        def done(combo: str) -> None:
            def apply() -> None:
                self.listen_hotkey_entry.delete(0, "end")
                self.listen_hotkey_entry.insert(0, combo)
                self.listen_record_btn.configure(text="⌨  Ghi tổ hợp phím")
                self._toast("Đã ghi phím tắt nghe — bấm Lưu để áp dụng.")
                self._start_listen_hotkey()
            self._post(apply)
        self.recorder = HotkeyRecorder(done)
        self.recorder.start()

    def _start_listen_hotkey(self) -> None:
        self.listen_hotkeys.stop()
        combo = (self.config_data.get("listen_hotkey") or "").strip()
        if not combo:
            return
        try:
            self.listen_hotkeys.start(combo)
        except Exception as exc:
            print("Listen hotkey error:", exc)

    def _record_service_hotkey(self) -> None:
        self.service_record_btn.configure(text="Đang nghe…")
        self.service_hotkeys.stop()
        def done(combo: str) -> None:
            def apply() -> None:
                self.service_hotkey_entry.delete(0, "end")
                self.service_hotkey_entry.insert(0, combo)
                self.service_record_btn.configure(text="⌨  Ghi tổ hợp phím")
                self._toast("Đã ghi phím tắt dịch vụ — bấm Lưu để áp dụng.")
                self._start_service_hotkey()
            self._post(apply)
        self.recorder = HotkeyRecorder(done)
        self.recorder.start()

    def _start_service_hotkey(self) -> None:
        self.service_hotkeys.stop()
        combo = (self.config_data.get("service_hotkey") or "").strip()
        if not combo:
            return
        try:
            self.service_hotkeys.start(combo)
        except Exception as exc:
            print("Service hotkey error:", exc)

    # ------------------------- System Tray ------------------------- #
    def _setup_app_icon(self) -> None:
        try:
            self._icon_pil = load_app_icon(256)
            ico_path = BASE_DIR / "app_icon.ico"
            self._icon_pil.save(ico_path, format="ICO",
                                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
            self._icon_photo = ImageTk.PhotoImage(self._icon_pil.resize((64, 64), Image.LANCZOS))
            self.iconphoto(True, self._icon_photo)
            if sys.platform == "win32":
                # customtkinter tự đặt icon mặc định sau ~200ms, nên ta đặt lại sau đó.
                def apply() -> None:
                    try:
                        self.iconbitmap(str(ico_path))
                    except Exception:
                        pass
                self.after(350, apply)
        except Exception as exc:
            print("Icon error:", exc)
            self._icon_pil = None

    def _tray_image(self) -> Image.Image:
        if getattr(self, "_icon_pil", None) is not None:
            return self._icon_pil.resize((64, 64), Image.LANCZOS)
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((2, 2, 62, 62), fill=PALETTE.get("primary", "#2563eb") if str(PALETTE.get("primary", "")).startswith("#") else "#2563eb")
        d.text((22, 16), "Q", fill="white")
        d.rectangle((20, 40, 44, 46), fill="white")
        return img

    def _start_tray(self) -> None:
        if pystray is None:
            print("Chưa cài pystray: pip install pystray pillow")
            return
        menu = pystray.Menu(
            pystray.MenuItem("Hiện ứng dụng", lambda *_: self._post(self._show_from_tray), default=True),
            pystray.MenuItem(lambda _i: "Tắt nghe" if self.listener_active() else "Bật nghe",
                             lambda *_: self._post(self._toggle_listen)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Thoát", lambda *_: self._post(self._quit_app)),
        )
        self.tray_icon = pystray.Icon("StudyQuestionTracker", self._tray_image(),
                                      "Study Question Tracker", menu)
        try:
            self.tray_icon.run_detached()
        except Exception:
            threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def listener_active(self) -> bool:
        return bool(getattr(self.listener, "active", False) or getattr(self.listener, "running", False))

    def _hide_to_tray(self) -> None:
        if self.tray_icon is None:
            self.iconify()
            return
        self.toast.hide()
        self.withdraw()  # ẩn cửa sổ và biến mất khỏi Taskbar
        try:
            self.tray_icon.notify("Ứng dụng vẫn chạy ở khay hệ thống.", "Study Question Tracker")
        except Exception:
            pass

    def _show_from_tray(self) -> None:
        self.deiconify()
        self.state("normal")
        self._bring_to_front()

    def _tray_enabled(self) -> bool:
        return self.tray_icon is not None and bool(self.config_data.get("minimize_to_tray", True))

    def _on_unmap(self, event) -> None:
        if event.widget is self and not self._quitting and self._tray_enabled():
            safe_after(self, 10, lambda: self.state() == "iconic" and self._hide_to_tray())

    def _on_window_close(self) -> None:
        """Bấm X = thoát hẳn ứng dụng. Chỉ nút thu nhỏ (–) mới ẩn xuống khay chạy ngầm."""
        self._quit_app()

    def _quit_app(self) -> None:
        self._quitting = True
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
            self.tray_icon = None
        self._on_close()
        # Chốt chặn: nếu còn luồng nền nào giữ tiến trình, tự kết thúc sau 1.5 giây.
        guard = threading.Timer(1.5, lambda: os._exit(0))
        guard.daemon = True
        guard.start()

    def _mode_changed(self) -> None:
        self.config_data["scan_mode"] = self.mode_var.get()
        self._persist_config()

    def _pick_region(self) -> None:
        self.withdraw()
        safe_after(self, 250, lambda: RegionSelector(self, self._region_done))

    def _region_done(self, region: list[int] | None, _img=None) -> None:
        self.deiconify()
        if region:
            self.config_data["region"] = region
            self.config_data["scan_mode"] = "fixed"
            self.mode_var.set("fixed")
            self.region_lbl.configure(text=self._region_text())
            self._persist_config()

    def _bring_to_front(self) -> None:
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        safe_after(self, 200, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def _apply_ai_settings(self) -> None:
        """Đẩy tốc độ + danh sách model dự phòng từ cấu hình vào bộ gọi AI (gọi trước mỗi lượt quét/chat)."""
        self.gemini.speed_mode = str(self.config_data.get("ai_speed", "fast"))
        try:
            self.gemini.rpm_limit = max(0, int(self.config_data.get("rpm_limit", 5)))
        except Exception:
            self.gemini.rpm_limit = 5
        if self.config_data.get("auto_fallback", True):
            chain = self.config_data.get("model_fallbacks") or DEFAULT_FALLBACKS
            self.gemini.fallbacks = [m for m in chain if m in MODEL_CHOICES]
        else:
            self.gemini.fallbacks = []

    def _on_model_switch(self, old: str, new: str, daily: bool) -> None:
        reason = "đã hết lượt miễn phí hôm nay" if daily else "đang quá tải/vượt giới hạn"
        self._post(lambda: (
            self.toast.show(f"🔁 Chuyển sang {new}", f"{old} {reason}.", PALETTE["warning"], 6000),
            self._refresh_quota_label(),
        ))

    def _refresh_quota_label(self) -> None:
        text = self.gemini.quota.summary() if self.gemini.quota else ""
        for name in ("quota_lbl", "quota_settings_lbl"):
            lbl = getattr(self, name, None)
            if lbl is not None:
                try:
                    lbl.configure(text=text if name == "quota_settings_lbl" else
                                  "Lượt AI hôm nay:\n" + text.replace("  ⛔ hết lượt", " ⛔"))
                except Exception:
                    pass

    def _write_config(self) -> None:
        """Ghi config.json. Khi có kho mật khẩu, API key chỉ nằm trong kho chứ không ghi ra file."""
        data = dict(self.config_data)
        key = str(data.get("api_key") or "")
        if self._vault_ok and key != self._vault_key:
            if secret_set(key):
                self._vault_key = key
            else:
                self._vault_ok = False
        if self._vault_ok:
            data.pop("api_key", None)
        save_json(CONFIG_PATH, data)

    def _api_status_text(self) -> tuple[str, str]:
        if self._vault_ok:
            where = "Windows Credential Manager" if sys.platform == "win32" else "kho mật khẩu của hệ điều hành"
            return f"🔒 Key được lưu an toàn trong {where}, không nằm trong config.json.", PALETTE["muted"]
        return ("⚠ Không dùng được kho mật khẩu: key đang nằm trong config.json "
                "(file này đã được thêm vào .gitignore)."), PALETTE["warning"]

    def _persist_config(self) -> None:
        self._write_config()

    def _save_settings(self) -> None:
        new_hotkey = self.hotkey_entry.get().strip() or DEFAULT_CONFIG["hotkey"]
        try:
            parse_hotkey(new_hotkey)
        except Exception:
            self._toast("Tổ hợp phím không hợp lệ!", ok=False)
            return
        new_listen_hotkey = self.listen_hotkey_entry.get().strip()
        if new_listen_hotkey:
            try:
                parse_hotkey(new_listen_hotkey)
            except Exception:
                self._toast("Phím tắt nghe không hợp lệ!", ok=False)
                return
            if parse_hotkey(new_listen_hotkey) == parse_hotkey(new_hotkey):
                self._toast("Phím tắt nghe phải khác phím tắt quét!", ok=False)
                return
        new_service_hotkey = self.service_hotkey_entry.get().strip()
        if new_service_hotkey:
            try:
                parse_hotkey(new_service_hotkey)
            except Exception:
                self._toast("Phím tắt bật/tắt dịch vụ không hợp lệ!", ok=False)
                return
            if parse_hotkey(new_service_hotkey) == parse_hotkey(new_hotkey) or (
                new_listen_hotkey and parse_hotkey(new_service_hotkey) == parse_hotkey(new_listen_hotkey)
            ):
                self._toast("Phím tắt dịch vụ phải khác các phím tắt còn lại!", ok=False)
                return
        new_model = normalize_model(self.model_menu.get())
        if new_model != self.model_menu.get():
            self.model_menu.set(new_model)
        self.config_data.update({
            "api_key": self.api_entry.get().strip(),
            "model": new_model,
            "ai_speed": "accurate" if "Chính xác" in self.speed_menu.get() else "fast",
            "auto_fallback": bool(self.fallback_var.get()),
            "rpm_limit": int(self.rpm_entry.get()) if self.rpm_entry.get().strip().isdigit() else 5,
            "model_fallbacks": [m.strip() for m in self.fallback_entry.get().split(",")
                                if m.strip() in MODEL_CHOICES] or list(DEFAULT_FALLBACKS),
            "hotkey": new_hotkey,
            "service_hotkey": new_service_hotkey,
            "listen_hotkey": new_listen_hotkey,
            "minimize_to_tray": bool(self.tray_var.get()),
            "save_screenshots": bool(self.save_shots_var.get()),
            "scan_mode": self.mode_var.get(),
            "mouse_trigger": bool(self.mouse_var.get()),
            "mouse_button": next((k for k, v in MOUSE_BUTTONS.items() if v == self.mouse_menu.get()), "right"),
            "mouse_clicks": next((k for k, v in MOUSE_CLICKS.items() if v == self.mouse_clicks_menu.get()), 2),
        })
        self._persist_config()
        status, status_color = self._api_status_text()
        self.api_status.configure(text=status, text_color=status_color)
        self._start_listen_hotkey()
        self._start_service_hotkey()
        self._update_hotkey_label()
        if self.hotkeys.active:
            try:
                self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
                self.hotkeys.start(new_hotkey)
                self._sync_mouse()
            except Exception as exc:
                self._set_service_ui(False)
                self._toast(f"Đã lưu nhưng dịch vụ tắt: {exc}", ok=False)
                return
        self._toast("Đã lưu cấu hình thành công.")

    def _on_hotkey(self) -> None:
        if self._busy:
            self.toast.show("Đang phân tích câu trước…", "Vui lòng đợi giây lát.", PALETTE["warning"])
            return
        try:
            self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
        except Exception as exc:
            self.toast.show("Không quét được", str(exc), PALETTE["danger"], 6000)
            self.tabs.set("Cài đặt")
            self._bring_to_front()
            return
        self._busy = True
        self._was_visible = self.state() not in ("withdrawn", "iconic")
        mode = self.config_data.get("scan_mode", "select")
        region = self.config_data.get("region")
        if mode == "fixed" and not region:
            mode = "select"
        self.withdraw()
        self.toast.hide()
        if mode == "select":
            safe_after(self, 220, lambda: RegionSelector(self, self._on_region_selected))
        else:
            safe_after(self, 220, lambda: self._capture_now(region if mode == "fixed" else None))

    def _on_region_selected(self, region, img) -> None:
        if not region or img is None:
            self._busy = False
            self._restore_window()
            self.toast.show("Đã hủy quét", "", PALETTE["muted"], 1800)
            return
        self._after_capture(img, region)

    def _capture_now(self, region: list[int] | None) -> None:
        try:
            image = grab_screen(region)
        except Exception as exc:
            self._busy = False
            self._restore_window()
            self.toast.show("Lỗi chụp màn hình", str(exc), PALETTE["danger"], 6000)
            return
        self._after_capture(image, region)

    def _after_capture(self, image: Image.Image, region: list[int] | None) -> None:
        flash_screen(self, region)
        self._restore_window()
        self.status_lbl.configure(text="Gemini đang phân tích…")
        self.status_dot.configure(text_color=PALETTE["warning"])
        self.toast.show("📸 Đã chụp màn hình", "Gemini đang phân tích câu hỏi…", PALETTE["primary"], timeout_ms=None)
        listen_text, audio_clips, upto, seconds = self._take_listen_payload()
        if listen_text or audio_clips:
            detail = (f"Gửi kèm {len(audio_clips)} đoạn nghe (~{seconds:.0f}s) trong 1 yêu cầu duy nhất…"
                      if audio_clips else "Gemini đang phân tích ảnh và đoạn vừa nghe…")
            self.toast.show("📸 Đã chụp + 🎧 kèm bài nghe", detail, PALETTE["primary"], timeout_ms=None)
        threading.Thread(target=self._analyze_worker,
                         args=(image, listen_text, audio_clips, upto), daemon=True).start()

    def _restore_window(self) -> None:
        if getattr(self, "_was_visible", True):
            self.deiconify()
        else:
            self.iconify()

    def _analyze_worker(self, image: Image.Image, listen_text: str = "",
                        audio_clips: list[bytes] | None = None, upto: float = 0.0) -> None:
        def on_retry(wait: float, attempt: int) -> None:
            if attempt == 0:
                self._post(lambda: self.toast.show(
                    "⏱ Giữ nhịp lượt/phút của gói miễn phí",
                    f"Đợi {wait:.0f} giây để không bị khóa tạm thời (đỡ tốn lượt).",
                    PALETTE["primary"], int(wait * 1000) + 1500))
                return
            self._post(lambda: self.toast.show(
                "⏳ Đang vượt quá giới hạn API",
                f"Vui lòng đợi {wait:.0f} giây… (tự thử lại lần {attempt})",
                PALETTE["warning"], int(wait * 1000) + 1500))

        try:
            self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
            self._apply_ai_settings()
            shot_path = ""
            if self.config_data.get("save_screenshots"):
                stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = str(SHOTS_DIR / f"shot_{stamp}.png")
                # Lưu ảnh ở luồng riêng, song song với lúc chờ Gemini (không làm chậm đáp án).
                threading.Thread(target=lambda: image.save(shot_path), daemon=True).start()
            answer = self.gemini.solve_image(image, listen_text, audio_clips, on_retry=on_retry)
            entry = {
                "id": uuid.uuid4().hex,
                "time": _dt.datetime.now().strftime("%H:%M:%S"),
                "answer": answer,
                "screenshot": shot_path,
                "review": _default_review_state(),
            }
            if listen_text or audio_clips:
                entry["source"] = "screen+audio"
                if listen_text:
                    entry["listen_text"] = listen_text
            if listen_text or audio_clips:
                n = len(audio_clips or [])
                self._post(lambda: self._mark_listen_log_used(upto, listen_text, n))
            self._post(lambda: self._finish_scan(entry, answer))
        except ApiBusyError as exc:
            message = str(exc)
            self._api_busy_until = time.time() + 15
            self._post(lambda: self.toast.show("⏳ Đang vượt quá giới hạn API",
                                               message[:300], PALETTE["warning"], 10000))
        except Exception as exc:
            message = friendly_api_error(exc)
            print("Lỗi phân tích:", type(exc).__name__, message[:200])
            self._post(lambda: self.toast.show("Lỗi phân tích", message[:300], PALETTE["danger"], 8000))
        finally:
            self._busy = False
            self._post(lambda: self._set_service_ui(self.hotkeys.active))

    def _find_duplicate(self, entry: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        sig = question_signature(str(entry.get("answer", "")))
        if not sig:
            return None
        for day, items in self.history.items():
            for item in items:
                if question_signature(str(item.get("answer", ""))) == sig:
                    return day, item
        return None

    def _add_entry(self, entry: dict[str, Any]) -> bool:
        """Thêm bản ghi mới; nếu câu hỏi đã có thì gộp vào bản cũ. Trả về True nếu là bản trùng."""
        found = self._find_duplicate(entry)
        if found is not None:
            old_day, old = found
            _merge_entry_into(old, {**entry, "review": {}})
            old["last_scanned_at"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_json(HISTORY_PATH, self.history)
            self._refresh_all(old_day)
            self._show_day(old_day)
            self.tabs.set("Lịch sử đã quét")
            self.toast.show("🔁 Câu này đã quét rồi — đã gộp",
                            f"Quét lần thứ {old['scan_count']} • giữ nguyên bản ngày {old_day}.",
                            PALETTE["primary"], 4500)
            return True
        day = _dt.date.today().isoformat()
        entry.setdefault("is_starred", False)
        self.history.setdefault(day, []).append(entry)
        save_json(HISTORY_PATH, self.history)
        self._refresh_all(day)
        self._show_day(day)
        self.tabs.set("Lịch sử đã quét")
        return False

    def _finish_scan(self, entry: dict[str, Any], answer: str) -> None:
        self._refresh_quota_label()
        if not self._add_entry(entry):
            self.toast.show("✓ Đã có đáp án — bấm để xem", _short_answer(answer), PALETTE["success"], 5500)

    def _merge_duplicates_clicked(self) -> None:
        total = sum(len(v) for v in self.history.values())
        merged_history, merged = merge_duplicate_history(self.history)
        if merged == 0:
            self.toast.show("Không có câu trùng", f"Đã kiểm tra {total} câu hỏi.", PALETTE["success"], 3500)
            return
        if not self._confirm(f"Tìm thấy {merged} câu hỏi trùng. Gộp chúng lại? (kết quả làm bài và dấu sao được giữ)"):
            return
        self.history = merged_history
        save_json(HISTORY_PATH, self.history)
        self.current_day = None
        self._refresh_all()
        self.toast.show("🧹 Đã gộp câu trùng", f"Đã gộp {merged} câu, còn {total - merged} câu.", PALETTE["success"], 4500)

    def _refresh_all(self, preferred_quiz_source: str | None = None) -> None:
        self._refresh_days()
        self.quiz_panel.refresh_sources(preferred_quiz_source)
        self.review_panel.refresh_sources()
        count = _review_count(self.history)
        self.review_count_label.configure(text=f"Cần ôn tập: {count} câu" if count else "Không có câu cần ôn")

    def _refresh_days(self) -> None:
        for widget in self.day_list.winfo_children():
            widget.destroy()
        days = sorted(self.history.keys(), reverse=True)
        total = sum(len(items) for items in self.history.values())
        self.sidebar_summary.configure(text=f"{total} câu hỏi")
        for day in days:
            items = self.history[day]
            review_count = sum(1 for item in items if item.get("review", {}).get("needs_review"))
            suffix = f"  •  {review_count} cần ôn" if review_count else ""
            self._button(
                self.day_list, f"▦  {day}  ({len(items)}){suffix}",
                lambda selected=day: self._show_day(selected), kind="neutral",
                anchor="w", height=40,
            ).pack(fill="x", pady=4)
        if not days:
            ctk.CTkLabel(self.day_list, text="Chưa có lịch sử", text_color=PALETTE["muted"]).pack(pady=18)

    def _show_day(self, day: str) -> None:
        self.current_day = day
        items = self.history.get(day, [])
        self.day_title.configure(text=f"Ngày {day}  •  {len(items)} câu hỏi")
        for widget in self.results_box.winfo_children():
            widget.destroy()
        for index, item in enumerate(reversed(items), start=1):
            parsed = parse_quiz_content(str(item.get("answer", "")))
            review = item.get("review", {})
            card = ctk.CTkFrame(
                self.results_box, fg_color=PALETTE["surface"], corner_radius=12,
                border_width=1, border_color=PALETTE["border"],
            )
            card.pack(fill="x", pady=7, padx=4)
            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=16, pady=(13, 5))
            ctk.CTkLabel(
                header, text=f"Câu {index}  •  {item.get('time', '')}",
                text_color=PALETTE["blue"], font=ctk.CTkFont(size=12, weight="bold"),
            ).pack(side="left")
            ctk.CTkButton(
                header, text="🗑 Xóa mục này", width=104, height=26, corner_radius=7,
                fg_color=PALETTE["surface_2"], hover_color=PALETTE["danger_soft"],
                text_color=PALETTE["danger"], font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda it=item, n=index: self._delete_history_item(day, it, n),
            ).pack(side="right", padx=(8, 0))
            starred = bool(item.get("is_starred", False))
            ctk.CTkButton(
                header, text="★" if starred else "☆", width=30, height=26, corner_radius=7,
                fg_color="transparent", hover_color=PALETTE["surface_hover"],
                border_width=1,
                border_color=PALETTE["warning"] if starred else PALETTE["border"],
                text_color=PALETTE["warning"] if starred else PALETTE["muted"],
                font=ctk.CTkFont(size=16),
                command=lambda d=day, it=item: self.toggle_star(d, it),
            ).pack(side="right")
            if review.get("needs_review"):
                ctk.CTkLabel(
                    header, text="CẦN ÔN", fg_color=PALETTE["danger_soft"],
                    text_color=PALETTE["danger"], corner_radius=7, padx=9, pady=3,
                    font=ctk.CTkFont(size=10, weight="bold"),
                ).pack(side="right")
            ctk.CTkLabel(
                card, text=parsed["question"], anchor="w", justify="left", wraplength=800,
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(fill="x", padx=16, pady=(2, 8))
            body = ctk.CTkTextbox(
                card, wrap="word", height=200, fg_color=PALETTE["surface_2"],
                border_width=0, corner_radius=8, font=ctk.CTkFont(size=12),
            )
            body.pack(fill="x", padx=16, pady=(0, 14))
            body.insert("end", str(item.get("answer", "")))
            body.configure(state="disabled")

    def get_review_entries(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (day, item) for day in sorted(self.history.keys(), reverse=True)
            for item in self.history[day]
            if item.get("review", {}).get("needs_review", False)
        ]

    def get_starred_entries(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (day, item) for day in sorted(self.history.keys(), reverse=True)
            for item in self.history[day]
            if bool(item.get("is_starred", False))
        ]

    def _setup_global_keybindings(self) -> None:
        """Đăng ký phím toàn cục một lần khi khởi động (Space = Câu tiếp, cuộn Cài đặt)."""
        self.bind_all("<KeyPress-space>", self._on_space_next)
        self.bind_all("<Key>", self._on_quiz_key, add="+")
        self.smooth_scroller = SmoothScroller(self)
        if getattr(self, "settings_scroll", None) is not None:
            self._bind_scroll_wheel(self.settings_scroll)

    def _on_space_next(self, event=None, from_panel: bool = False):
        """Phím Space = 'Câu tiếp' sau khi đã trả lời, bỏ qua khi đang gõ trong ô nhập."""
        try:
            tab = self.tabs.get()
        except Exception:
            return None
        panel = {"Ôn tập & Làm bài": self.quiz_panel, "Cần ôn tập": self.review_panel}.get(tab)
        if panel is None:
            return None
        if not from_panel:
            try:
                widget = self.focus_get()
            except Exception:
                widget = None
            if widget is not None and widget.winfo_class() in ("Entry", "TEntry", "Text", "Spinbox", "TCombobox"):
                inside_panel = False
                w = widget
                while w is not None:
                    if w is panel:
                        inside_panel = True
                        break
                    w = getattr(w, "master", None)
                editable = str(widget.cget("state")) not in ("disabled", "readonly") if "state" in widget.keys() else True
                if not inside_panel and editable and widget.winfo_viewable():
                    return None
        if not panel.entries:
            return "break"
        if panel.answered_current and str(panel.next_btn.cget("state")) == "normal":
            panel.next()
        return "break"

    def _focus_is_typing(self) -> bool:
        try:
            widget = self.focus_get()
        except Exception:
            return False
        if widget is None or widget.winfo_class() not in ("Entry", "TEntry", "Text", "Spinbox", "TCombobox"):
            return False
        try:
            state = str(widget.cget("state")) if "state" in widget.keys() else "normal"
        except Exception:
            state = "normal"
        return state not in ("disabled", "readonly")

    def _on_quiz_key(self, event):
        """Phím tắt làm bài: A–D / 1–4 chọn đáp án, ←/→ chuyển câu (chỉ ở tab làm bài/ôn tập, không khi đang gõ)."""
        if event.state & 0x4 or event.state & 0x20000 or event.state & 0x8:   # Ctrl / Alt
            return None
        try:
            tab = self.tabs.get()
        except Exception:
            return None
        panel = {"Ôn tập & Làm bài": self.quiz_panel, "Cần ôn tập": self.review_panel}.get(tab)
        if panel is None or self._focus_is_typing():
            return None
        return "break" if panel.handle_key(event.keysym) else None

    def toggle_star(self, day: str, entry: dict[str, Any]) -> None:
        """Đổi dấu sao độc lập với kết quả làm bài và lưu ngay vào lịch sử."""
        if entry not in self.history.get(day, []):
            return
        entry["is_starred"] = not bool(entry.get("is_starred", False))
        save_json(HISTORY_PATH, self.history)
        entry_id = str(entry.get("id", ""))
        self.quiz_panel.sync_star_state(entry_id)
        self.review_panel.sync_star_state(entry_id)
        if self.review_panel.review_mode == "starred":
            self.review_panel.answered_current = False
            self.review_panel.refresh_sources()
        else:
            self.review_panel._update_review_counts()
        if self.current_day == day:
            self._show_day(day)
        state = "Đã gắn sao" if entry["is_starred"] else "Đã bỏ dấu sao"
        self.toast.show(state, "Trạng thái câu hỏi đã được lưu.", PALETTE["warning"], 2500)

    def record_attempt(self, day: str, entry: dict[str, Any], chosen: str, is_correct: bool) -> None:
        review = {**_default_review_state(), **entry.get("review", {})}
        now = _dt.datetime.now().isoformat(timespec="seconds")
        review["attempts"] = int(review.get("attempts", 0) or 0) + 1
        review["wrong_count"] = int(review.get("wrong_count", 0) or 0) + (0 if is_correct else 1)
        review["last_choice"] = chosen
        review["last_correct"] = is_correct
        review["last_attempt_at"] = now
        review["needs_review"] = not is_correct
        attempts = review.get("attempt_history", [])
        if not isinstance(attempts, list):
            attempts = []
        attempts.append({"choice": chosen, "correct": is_correct, "at": now})
        review["attempt_history"] = attempts[-100:]
        entry["review"] = review
        save_json(HISTORY_PATH, self.history)
        self._refresh_days()
        count = _review_count(self.history)
        self.review_count_label.configure(text=f"Cần ôn tập: {count} câu" if count else "Không có câu cần ôn")
        if self.review_panel.review_mode == "starred":
            # Câu gắn sao luôn giữ nguyên dù đúng hay sai — chỉ cập nhật số lượng.
            self.review_panel._update_review_counts()
        elif is_correct:
            # Giữ phản hồi đúng trong chốc lát rồi gỡ câu khỏi danh sách cần ôn.
            def _drop_correct() -> None:
                self.review_panel.answered_current = False
                self.review_panel.refresh_sources()
            safe_after(self.review_panel, 900, _drop_correct)
        else:
            self.review_panel.refresh_sources()

    def _delete_history_item(self, day: str, item: dict[str, Any], number: int) -> None:
        items = self.history.get(day, [])
        if item not in items:
            return
        if not self._confirm(f"Xóa Câu {number} lúc {item.get('time', '')} ngày {day}?"):
            return
        items.remove(item)
        self._after_history_delete(day)

    def _after_history_delete(self, day: str) -> None:
        if not self.history.get(day):
            self.history.pop(day, None)
            save_json(HISTORY_PATH, self.history)
            self.current_day = None
            self.day_title.configure(text="Chọn một ngày ở bên trái")
            for widget in self.results_box.winfo_children():
                widget.destroy()
            self._refresh_all()
            return
        save_json(HISTORY_PATH, self.history)
        self._refresh_all()
        self._show_day(day)

    def _ask_time_range(self, day: str) -> tuple[str, str] | None:
        """Hộp chọn khung giờ bằng menu thả xuống (giờ/phút), trả về (HH:MM:SS, HH:MM:SS)."""
        result: dict[str, tuple[str, str] | None] = {"range": None}
        dialog = ctk.CTkToplevel(self)
        dialog.title("Xóa theo khung giờ")
        dialog.geometry("360x230")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        ctk.CTkLabel(dialog, text=f"Ngày {day} — chọn khung giờ cần xóa:",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(16, 10))
        hours = [f"{h:02d}" for h in range(24)]
        minutes = [f"{m:02d}" for m in range(0, 60, 5)]
        row1 = ctk.CTkFrame(dialog, fg_color="transparent")
        row1.pack(pady=4)
        ctk.CTkLabel(row1, text="Từ:", width=36, anchor="e").pack(side="left", padx=(0, 6))
        from_h = ctk.CTkOptionMenu(row1, values=hours, width=76)
        from_h.set("08")
        from_h.pack(side="left")
        ctk.CTkLabel(row1, text=":").pack(side="left", padx=2)
        from_m = ctk.CTkOptionMenu(row1, values=minutes, width=76)
        from_m.set("00")
        from_m.pack(side="left")
        row2 = ctk.CTkFrame(dialog, fg_color="transparent")
        row2.pack(pady=4)
        ctk.CTkLabel(row2, text="Đến:", width=36, anchor="e").pack(side="left", padx=(0, 6))
        to_h = ctk.CTkOptionMenu(row2, values=hours, width=76)
        to_h.set("09")
        to_h.pack(side="left")
        ctk.CTkLabel(row2, text=":").pack(side="left", padx=2)
        to_m = ctk.CTkOptionMenu(row2, values=minutes, width=76)
        to_m.set("00")
        to_m.pack(side="left")
        btns = ctk.CTkFrame(dialog, fg_color="transparent")
        btns.pack(pady=16)

        def ok() -> None:
            result["range"] = (f"{from_h.get()}:{from_m.get()}:00", f"{to_h.get()}:{to_m.get()}:59")
            dialog.destroy()

        self._button(btns, "✔ Chọn", ok, kind="primary", width=110, height=34).pack(side="left", padx=6)
        self._button(btns, "Hủy", dialog.destroy, kind="neutral", width=90, height=34).pack(side="left", padx=6)
        dialog.wait_window()
        return result["range"]

    def _delete_items_by_time(self) -> None:
        day = self.current_day
        if not day or day not in self.history:
            self.toast.show("Chưa chọn ngày", "Hãy chọn một ngày ở bên trái trước.", PALETTE["warning"], 4000)
            return
        picked = self._ask_time_range(day)
        if not picked:
            return
        start, end = picked
        if start > end:
            start, end = end, start
        items = self.history[day]
        matches = [it for it in items if start <= str(it.get("time", "")).zfill(8) <= end]
        if not matches:
            self.toast.show("Không có câu nào", f"Không có câu hỏi trong {start[:5]}–{end[:5]}.", PALETTE["muted"], 4000)
            return
        if not self._confirm(f"Xóa {len(matches)} câu hỏi trong khung {start[:5]}–{end[:5]} ngày {day}?"):
            return
        self.history[day] = [it for it in items if it not in matches]
        self._after_history_delete(day)

    def _delete_day(self) -> None:
        day = self.current_day
        if not day or day not in self.history:
            return
        del self.history[day]
        save_json(HISTORY_PATH, self.history)
        self.current_day = None
        self.day_title.configure(text="Chọn một ngày ở bên trái")
        for widget in self.results_box.winfo_children():
            widget.destroy()
        self._refresh_all()

    def _configure_chat_text(self) -> None:
        """Configure rich-text tags on the actual Tk Text inside CTkTextbox."""
        box = self.chat_view._textbox
        box.configure(padx=18, pady=14, spacing1=2, spacing3=3,
                      selectbackground=PALETTE["primary_hover"])
        box.tag_configure("speaker_ai", foreground=PALETTE["blue"],
                          font=(FONT_FAMILY, 12, "bold"), spacing1=9)
        # Tin của người dùng nằm bên PHẢI; trả lời của AI nằm bên trái như cũ.
        box.tag_configure("speaker_user", foreground="#7F8CFF", justify="right",
                          font=(FONT_FAMILY, 12, "bold"), spacing1=9)
        box.tag_configure("user_body", foreground=PALETTE["text"], font=(FONT_FAMILY, 13),
                          justify="right", spacing3=6, lmargin1=120, lmargin2=120)

        def fit_user_margin(event: Any) -> None:
            margin = max(60, int(event.width * 0.25))   # tin của bạn chiếm tối đa ~70% bề ngang
            box.tag_configure("user_body", lmargin1=margin, lmargin2=margin)
        box.bind("<Configure>", fit_user_margin, add="+")
        box.tag_configure("muted", foreground=PALETTE["muted"])
        box.tag_configure("body", foreground=PALETTE["text"], font=(FONT_FAMILY, 13),
                          spacing3=6)
        box.tag_configure("h1", foreground=PALETTE["text"],
                          font=(FONT_FAMILY, 20, "bold"), spacing1=13, spacing3=7)
        box.tag_configure("h2", foreground=PALETTE["text"],
                          font=(FONT_FAMILY, 17, "bold"), spacing1=11, spacing3=6)
        box.tag_configure("h3", foreground=PALETTE["blue"],
                          font=(FONT_FAMILY, 15, "bold"), spacing1=9, spacing3=5)
        box.tag_configure("bold", font=(FONT_FAMILY, 13, "bold"))
        box.tag_configure("italic", font=(FONT_FAMILY, 13, "italic"))
        box.tag_configure("bold_italic", font=(FONT_FAMILY, 13, "bold italic"))
        box.tag_configure("code_inline", foreground=PALETTE["blue"],
                          background=PALETTE["surface_2"], font=("Consolas", 12))
        box.tag_configure("code_block", foreground=PALETTE["text"],
                          background=PALETTE["surface_2"], font=("Consolas", 12),
                          lmargin1=16, lmargin2=16, spacing1=8, spacing3=8)
        box.tag_configure("rule", foreground=PALETTE["border"])
        for depth in range(4):
            box.tag_configure(f"list_{depth}", lmargin1=18 + depth * 22,
                              lmargin2=35 + depth * 22, spacing3=3)

    def _style_chat_entry_selection(self) -> None:
        """Ô nhập chat hoạt động như ô chat bình thường: gõ chữ không bị bôi đen.

        - Màu vùng chọn dịu hơn (chỉ hiện khi người dùng tự kéo chuột / Shift+mũi tên).
        - Tắt exportselection để vùng chọn ở khung hội thoại không "đè" sang ô nhập.
        - Sau mỗi ký tự gõ vào, tự bỏ vùng chọn còn sót lại.
        """
        inner = self.chat_entry._entry
        inner.configure(
            exportselection=False,
            selectbackground=PALETTE["primary_hover"],
            selectforeground="#FFFFFF",
        )
        inner.bind("<KeyRelease>", self._chat_entry_key_released, add="+")
        inner.bind("<FocusOut>", lambda _e: inner.selection_clear(), add="+")

    def _chat_entry_key_released(self, event=None) -> None:
        """Bỏ vùng chọn sau khi gõ ký tự thường (không đụng vào Shift/Ctrl + phím)."""
        if event is None:
            return
        if event.keysym in ("Shift_L", "Shift_R", "Control_L", "Control_R",
                            "Alt_L", "Alt_R", "Left", "Right", "Home", "End"):
            return
        inner = self.chat_entry._entry
        if event.state & 0x0005:  # Shift hoặc Ctrl đang giữ -> người dùng chủ động chọn/copy
            return
        try:
            if inner.selection_present():
                inner.selection_clear()
        except Exception:
            pass

    def _chat_entry_focused(self, _event=None) -> None:
        """Clear inherited selection without overriding the user's click position."""
        def settle() -> None:
            if self.chat_entry.focus_get() is self.chat_entry._entry:
                self.chat_entry.select_clear()
        self.chat_entry.after_idle(settle)

    def _reset_chat_entry_after_send(self) -> None:
        """Leave an empty, focused entry with a normal insertion cursor."""
        self.chat_entry.select_clear()
        self.chat_entry.icursor("end")
        self.chat_entry.focus_set()

        def settle() -> None:
            if self.chat_entry.focus_get() is self.chat_entry._entry:
                self.chat_entry.select_clear()
                self.chat_entry.icursor("end")
        self.chat_entry.after_idle(settle)

    def _send_chat_from_entry(self, _event=None) -> str:
        if self._chat_inflight is None:
            self._send_chat()
        return "break"

    def _on_send_or_stop(self) -> None:
        if self._chat_inflight is not None:
            self._stop_chat()
        else:
            self._send_chat()

    def _set_chat_busy(self, busy: bool) -> None:
        if busy:
            self.chat_send_btn.configure(text="■  Dừng", fg_color=PALETTE["danger_soft"],
                                         hover_color="#512033", text_color=PALETTE["danger"])
        else:
            self.chat_send_btn.configure(text="Gửi  →", fg_color=PALETTE["primary"],
                                         hover_color=PALETTE["primary_hover"], text_color=PALETTE["text"])

    def _insert_chat_inline(self, source: str, mark: str, style: str = "body") -> None:
        box = self.chat_view._textbox
        for value, inline_tag in _chat_inline_parts(source):
            if value:
                box.insert(mark, value, (style, inline_tag) if inline_tag else (style,))

    def _insert_chat_table(self, rows: list[list[str]], mark: str) -> None:
        box = self.chat_view._textbox
        table = tk.Frame(box, bg=PALETTE["border"], padx=1, pady=1)
        columns = len(rows[0])
        cell_width = max(65, min(210, (max(390, box.winfo_width() - 50) // columns) - 8))
        for col in range(columns):
            table.grid_columnconfigure(col, weight=1, uniform="chat_table")
        for row_index, row in enumerate(rows):
            for col, cell in enumerate(row):
                heading = row_index == 0
                bg = (PALETTE["surface_2"] if heading or row_index % 2 == 0
                      else PALETTE["surface"])
                cell_frame = tk.Frame(table, bg=bg)
                cell_frame.grid(row=row_index, column=col, sticky="nsew", padx=1, pady=1)
                cell_frame.grid_columnconfigure(0, weight=1)
                parts = _chat_inline_parts(cell)
                if all(not style for _text, style in parts):
                    label = tk.Label(cell_frame, text=cell, bg=bg,
                                     fg=PALETTE["blue"] if heading else PALETTE["text"],
                                     font=(FONT_FAMILY, 11, "bold" if heading else "normal"),
                                     wraplength=cell_width - 18, anchor="w", justify="left",
                                     padx=9, pady=7)
                    label.grid(row=0, column=0, sticky="nsew")
                else:
                    rich = tk.Text(cell_frame, wrap="word", width=16, height=1,
                                   bg=bg, fg=PALETTE["blue"] if heading else PALETTE["text"],
                                   font=(FONT_FAMILY, 11, "bold" if heading else "normal"),
                                   padx=9, pady=7, relief="flat", borderwidth=0,
                                   highlightthickness=0, cursor="arrow")
                    rich.tag_configure("bold", font=(FONT_FAMILY, 11, "bold"))
                    rich.tag_configure("italic", font=(FONT_FAMILY, 11, "italic"))
                    rich.tag_configure("bold_italic", font=(FONT_FAMILY, 11, "bold italic"))
                    rich.tag_configure("code_inline", font=("Consolas", 10), foreground=PALETTE["blue"])
                    for value, style in parts:
                        rich.insert("end", value, style or ())
                    rich.configure(state="disabled")
                    rich.grid(row=0, column=0, sticky="nsew")
                    def fit_cell(_event=None, widget=rich) -> None:
                        widget.configure(height=max(1, min(12, int(widget.count("1.0", "end", "displaylines")[0]))))
                    rich.bind("<Configure>", fit_cell)
        box.window_create(mark, window=table, align="baseline")
        box.insert(mark, "\n")

    def _insert_chat_message(self, who: str, text: str, index: str = "end-1c",
                             actions: str = "") -> tuple[str, str]:
        """Insert one message, returning its stable Tk index range for pending replacement."""
        box = self.chat_view._textbox
        mark = f"chat_insert_{uuid.uuid4().hex}"
        box.mark_set(mark, index)
        box.mark_gravity(mark, "right")
        start = box.index(mark)
        box.insert(mark, f"{who}\n", "speaker_user" if who == "Bạn" else "speaker_ai")
        if who == "AI" and text != "Đang soạn câu trả lời…":
            for kind, content in _chat_blocks(text):
                if kind == "table":
                    self._insert_chat_table(content, mark)
                elif kind == "list":
                    for level, bullet, item in content:
                        box.insert(mark, f"{bullet if bullet[0].isdigit() else '•'}  ", f"list_{level}")
                        self._insert_chat_inline(item, mark, f"list_{level}")
                        box.insert(mark, "\n", f"list_{level}")
                elif kind == "rule":
                    box.insert(mark, "────────────────────────\n", "rule")
                elif kind == "code":
                    box.insert(mark, f"{content}\n", "code_block")
                else:
                    self._insert_chat_inline(content, mark, kind if kind.startswith("h") else "body")
                    box.insert(mark, "\n", kind if kind.startswith("h") else "body")
        else:
            box.insert(mark, f"{text}\n", "muted" if who == "AI" else "user_body")
        if who == "AI" and actions:
            self._insert_chat_actions(mark, text, actions)
        box.insert(mark, "\n")
        end = box.index(mark)
        box.mark_unset(mark)
        box.see(end)
        return start, end

    def _append_chat(self, who: str, text: str) -> tuple[str, str]:
        self.chat_view.configure(state="normal")
        span = self._insert_chat_message(who, text)
        self.chat_view.configure(state="disabled")
        return span

    # ----- Lịch sử chat AI: lưu nhiều cuộc trò chuyện vào chat_history.json -----
    def _chat_sorted(self) -> list[dict[str, Any]]:
        convs = [c for c in self._chat_store["conversations"] if isinstance(c, dict)]
        return sorted(convs, key=lambda c: c.get("updated", 0), reverse=True)

    def _active_chat(self) -> dict[str, Any] | None:
        for conv in self._chat_store["conversations"]:
            if isinstance(conv, dict) and conv.get("id") == self._chat_active_id:
                return conv
        return None

    def _save_chat_store(self) -> None:
        self._chat_store["active"] = self._chat_active_id
        try:
            save_json(CHAT_HISTORY_PATH, self._chat_store)
        except Exception:
            pass

    def _create_chat(self) -> dict[str, Any]:
        now = time.time()
        conv = {"id": uuid.uuid4().hex, "title": "Chat mới", "created": now, "updated": now, "messages": []}
        self._chat_store["conversations"].append(conv)
        self._chat_active_id = conv["id"]
        self._save_chat_store()
        return conv

    def _refresh_chat_list(self) -> None:
        for widget in self.chat_list.winfo_children():
            widget.destroy()
        for conv in self._chat_sorted():
            active = conv["id"] == self._chat_active_id
            cid = conv["id"]
            title = str(conv.get("title") or "Chat mới")
            title = title if len(title) <= 20 else title[:19] + "…"
            row = ctk.CTkFrame(self.chat_list, fg_color="transparent")
            row.grid(sticky="ew", padx=2, pady=2)
            row.grid_columnconfigure(0, weight=1)
            fg = PALETTE["primary"] if active else "transparent"
            hover = PALETTE["primary_hover"] if active else PALETTE["surface_hover"]
            item = ctk.CTkButton(
                row, text=title, anchor="w", height=38, corner_radius=8, fg_color=fg, hover_color=hover,
                text_color="#EEF0FF" if active else PALETTE["text"], cursor="hand2",
                command=lambda c=cid: self._switch_chat(c),
            )
            item.grid(row=0, column=0, sticky="ew")
            more = ctk.CTkButton(
                row, text="⋯", width=32, height=38, corner_radius=8, fg_color=fg, hover_color=hover,
                text_color="#EEF0FF" if active else PALETTE["muted"], cursor="hand2",
                font=ctk.CTkFont(size=16, weight="bold"),
            )
            more.configure(command=lambda c=cid, b=more: self._chat_menu(c, b.winfo_rootx(),
                                                                         b.winfo_rooty() + b.winfo_height()))
            more.grid(row=0, column=1, padx=(2, 0))
            item.bind("<Button-3>", lambda e, c=cid: self._chat_menu(c, e.x_root, e.y_root), add="+")
        conv = self._active_chat()
        self.chat_title_label.configure(text=str(conv.get("title", "")) if conv else "")

    def _chat_menu(self, conv_id: str, x: int, y: int) -> None:
        """Menu ⋯ / chuột phải của từng cuộc trò chuyện: Đổi tên, Xóa."""
        menu = tk.Menu(self, tearoff=0, bg=PALETTE["surface_2"], fg=PALETTE["text"],
                       activebackground=PALETTE["primary"], activeforeground="#EEF0FF",
                       bd=0, relief="flat", font=(FONT_FAMILY, 11))
        menu.add_command(label="✎   Đổi tên", command=lambda: self._rename_chat(conv_id))
        menu.add_command(label="🗑   Xóa cuộc trò chuyện", command=lambda: self._delete_chat(conv_id))
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _prompt_text(self, title: str, label: str, initial: str = "") -> str | None:
        top = ctk.CTkToplevel(self)
        top.title(title)
        top.resizable(False, False)
        top.configure(fg_color=PALETTE["window"])
        width, height = 430, 178
        try:
            x = self.winfo_rootx() + (self.winfo_width() - width) // 2
            y = self.winfo_rooty() + (self.winfo_height() - height) // 3
            top.geometry(f"{width}x{height}+{max(x, 0)}+{max(y, 0)}")
            top.transient(self)
        except Exception:
            top.geometry(f"{width}x{height}")
        result: dict[str, str | None] = {"value": None}
        ctk.CTkLabel(top, text=label, anchor="w").pack(fill="x", padx=22, pady=(20, 6))
        entry = ctk.CTkEntry(top, height=40, fg_color=PALETTE["surface"], border_color=PALETTE["border"])
        entry.pack(fill="x", padx=22)
        entry.insert(0, initial)

        def ok(_event=None) -> None:
            result["value"] = entry.get().strip()
            top.destroy()

        def cancel(_event=None) -> None:
            top.destroy()

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=22, pady=18)
        self._button(row, "Lưu", ok, width=100, height=36).pack(side="right")
        self._button(row, "Hủy", cancel, kind="neutral", width=100, height=36).pack(side="right", padx=(0, 8))
        entry.bind("<Return>", ok)
        top.bind("<Escape>", cancel)

        def focus() -> None:
            try:
                top.grab_set()
                entry.focus_set()
                entry.select_range(0, "end")
            except Exception:
                pass
        top.after(150, focus)
        self.wait_window(top)
        return result["value"]

    def _rename_chat(self, conv_id: str) -> None:
        conv = next((c for c in self._chat_store["conversations"]
                     if isinstance(c, dict) and c.get("id") == conv_id), None)
        if conv is None:
            return
        new_title = self._prompt_text("Đổi tên cuộc trò chuyện", "Tên mới:", str(conv.get("title", "")))
        if not new_title:
            return
        conv["title"] = new_title[:60]
        self._save_chat_store()
        self._refresh_chat_list()

    def _delete_chat(self, conv_id: str) -> None:
        conv = next((c for c in self._chat_store["conversations"]
                     if isinstance(c, dict) and c.get("id") == conv_id), None)
        if conv is None:
            return
        if conv.get("messages") and not self._confirm(f"Xóa cuộc trò chuyện \"{conv.get('title', '')}\"?"):
            return
        self._chat_store["conversations"] = [
            c for c in self._chat_store["conversations"] if c is not conv]
        if self._chat_active_id == conv_id:
            remaining = self._chat_sorted()
            self._chat_active_id = remaining[0]["id"] if remaining else None
            self.gemini.reset_chat()
        self._save_chat_store()
        self._render_chat_view()
        self._refresh_chat_list()

    def _delete_all_chats(self) -> None:
        if not self._chat_store["conversations"]:
            return
        if not self._confirm("Xóa TẤT CẢ lịch sử chat với AI? Không thể khôi phục."):
            return
        self._chat_store["conversations"] = []
        self._chat_active_id = None
        self._save_chat_store()
        self.gemini.reset_chat()
        self._render_chat_view()
        self._refresh_chat_list()

    def _render_chat_view(self) -> None:
        conv = self._active_chat()
        self.chat_view.configure(state="normal")
        self.chat_view.delete("1.0", "end")
        if conv and conv.get("messages"):
            msgs = conv["messages"]
            for i, msg in enumerate(msgs):
                if msg.get("role") == "user":
                    self._insert_chat_message("Bạn", str(msg.get("text", "")))
                    continue
                last = i == len(msgs) - 1
                actions = "regen" if msg.get("error") else ("copy+regen" if last else "copy")
                if msg.get("error") and not last:
                    actions = ""
                self._insert_chat_message("AI", str(msg.get("text", "")), "end-1c", actions)
        else:
            self.chat_view.insert("end", "Trợ lý AI đã sẵn sàng giải đáp thắc mắc học tập của bạn.\n\n", "muted")
        self.chat_view.configure(state="disabled")
        self.chat_view._textbox.see("end")

    def _switch_chat(self, conv_id: str) -> None:
        if conv_id == self._chat_active_id:
            return
        self._chat_active_id = conv_id
        self._save_chat_store()
        self._render_chat_view()
        self._refresh_chat_list()
        self.chat_entry.focus_set()

    def _new_chat(self) -> None:
        conv = self._active_chat()
        if conv is None or conv.get("messages"):
            self._create_chat()
            self._render_chat_view()
            self._refresh_chat_list()
        self.chat_entry.focus_set()

    @staticmethod
    def _chat_seed(conv: dict[str, Any]) -> list[tuple[str, str]]:
        """Các cặp (người dùng, AI) đã trả lời thành công, dùng để AI nhớ ngữ cảnh khi mở lại cuộc trò chuyện."""
        msgs = conv.get("messages", [])
        seed: list[tuple[str, str]] = []
        i = 0
        while i < len(msgs):
            cur = msgs[i]
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if (cur.get("role") == "user" and nxt is not None and nxt.get("role") == "model"
                    and not nxt.get("error")):
                seed.append(("user", str(cur.get("text", ""))))
                seed.append(("model", str(nxt.get("text", ""))))
                i += 2
            else:
                i += 1
        return seed

    def _send_chat(self, message: str | None = None) -> None:
        if self._chat_inflight is not None:
            return
        if message is None:
            message = self.chat_entry.get().strip()
            if not message:
                return
            self.chat_entry.delete(0, "end")
            self._reset_chat_entry_after_send()
        conv = self._active_chat() or self._create_chat()
        seed = self._chat_seed(conv)
        if conv.get("title", "Chat mới") == "Chat mới":
            conv["title"] = " ".join(message.split())[:40] or "Chat mới"
        conv["messages"].append({"role": "user", "text": message})
        conv["updated"] = time.time()
        self._save_chat_store()
        self._refresh_chat_list()
        conv_id = conv["id"]

        self._append_chat("Bạn", message)
        start, end = self._append_chat("AI", "Đang soạn câu trả lời…")
        request_tag = f"chat_pending_{uuid.uuid4().hex}"
        self.chat_view._textbox.tag_add(request_tag, start, end)
        inflight: dict[str, Any] = {"tag": request_tag, "conv_id": conv_id, "cancelled": False}
        self._chat_inflight = inflight
        self._set_chat_busy(True)

        def work() -> None:
            failed = False
            try:
                self.gemini.configure(self.config_data["api_key"], self.config_data["model"])
                self._apply_ai_settings()

                def on_text(full: str) -> None:
                    if not inflight["cancelled"]:
                        self._post(lambda t=full: self._tw_push(request_tag, t))

                reply = self.gemini.chat(message, conv_id, seed, on_text, lambda: inflight["cancelled"])
            except ApiBusyError as exc:
                reply, failed = f"[⏳] {exc}", True
            except Exception as exc:
                reply, failed = f"[Lỗi] {friendly_api_error(exc)}", True
            if inflight["cancelled"]:
                return                      # bấm Dừng rồi: giao diện đã được xử lý ở _stop_chat

            def show() -> None:
                target = next((c for c in self._chat_store["conversations"]
                               if isinstance(c, dict) and c.get("id") == conv_id), None)
                if target is None:      # cuộc trò chuyện đã bị xóa trong lúc chờ
                    self._tw_cancel(request_tag)
                    self._chat_done(inflight)
                    return
                entry = {"role": "model", "text": reply}
                if failed:
                    entry["error"] = True
                target["messages"].append(entry)
                target["updated"] = time.time()
                self._save_chat_store()
                self._refresh_chat_list()

                def finalize() -> None:
                    self._chat_done(inflight)
                    self._replace_pending_bubble(request_tag, conv_id, reply, "regen" if failed else "copy+regen")

                self._tw_finish(request_tag, finalize, failed)
            self._post(show)
        threading.Thread(target=work, daemon=True).start()

    def _chat_done(self, inflight: dict[str, Any]) -> None:
        self._refresh_quota_label()
        if self._chat_inflight is inflight:
            self._chat_inflight = None
            self._set_chat_busy(False)

    def _replace_pending_bubble(self, tag: str, conv_id: str, text: str, actions: str = "") -> None:
        """Thay bong bóng "đang soạn/đang gõ" bằng tin AI hoàn chỉnh (có định dạng + nút thao tác)."""
        box = self.chat_view._textbox
        span = box.tag_ranges(tag)
        if len(span) == 2:
            self.chat_view.configure(state="normal")
            position = str(span[0])
            box.delete(span[0], span[1])
            box.tag_delete(tag)
            self._insert_chat_message("AI", text, position, actions)
            self.chat_view.configure(state="disabled")
        elif self._chat_active_id == conv_id:   # người dùng đã chuyển đi rồi quay lại
            self._render_chat_view()

    def _stop_chat(self) -> None:
        """Dừng câu trả lời đang viết; phần đã nhận được vẫn được giữ lại."""
        inflight = self._chat_inflight
        if inflight is None or inflight["cancelled"]:
            return
        inflight["cancelled"] = True
        tag, conv_id = inflight["tag"], inflight["conv_id"]
        state = self._typewriters.get(tag)
        partial = str(state["target"]).strip() if state else ""
        self._tw_cancel(tag)
        conv = next((c for c in self._chat_store["conversations"]
                     if isinstance(c, dict) and c.get("id") == conv_id), None)
        if conv is not None:
            if partial:
                text = partial + "\n\n*(đã dừng trả lời)*"
                conv["messages"].append({"role": "model", "text": text})
                conv["updated"] = time.time()
                self._save_chat_store()
                self._refresh_chat_list()
            else:
                text = "*(đã dừng trước khi AI trả lời)*"
            self._replace_pending_bubble(tag, conv_id, text, "regen")
        self.gemini.reset_chat()            # lần gửi sau dựng lại ngữ cảnh từ lịch sử đã lưu
        self._chat_done(inflight)

    def _regenerate_chat(self) -> None:
        """Hỏi lại: bỏ câu trả lời cuối của AI rồi gửi lại câu hỏi gần nhất của bạn."""
        if self._chat_inflight is not None:
            return
        conv = self._active_chat()
        if conv is None:
            return
        msgs = conv["messages"]
        if msgs and msgs[-1].get("role") == "model":
            msgs.pop()
        if not msgs or msgs[-1].get("role") != "user":
            return
        message = str(msgs.pop().get("text", ""))
        self._save_chat_store()
        self.gemini.reset_chat()
        self._render_chat_view()
        self._send_chat(message)

    def _copy_chat_text(self, text: str, button: Any) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            button.configure(text="✓ Đã chép")
            self.after(1500, lambda: button.winfo_exists() and button.configure(text="⧉ Sao chép"))
        except Exception:
            pass

    def _insert_chat_actions(self, mark: str, text: str, actions: str) -> None:
        """Hàng nút nhỏ dưới mỗi tin của AI: Sao chép / Hỏi lại (Hỏi lại chỉ ở tin cuối)."""
        box = self.chat_view._textbox
        bar = tk.Frame(box, bg=PALETTE["surface"])
        common = dict(height=26, corner_radius=7, fg_color="transparent", hover_color=PALETTE["surface_hover"],
                      text_color=PALETTE["muted"], border_width=1, border_color=PALETTE["border"],
                      font=ctk.CTkFont(size=11), cursor="hand2")
        if "copy" in actions:
            copy_btn = ctk.CTkButton(bar, text="⧉ Sao chép", width=96, **common)
            copy_btn.configure(command=lambda: self._copy_chat_text(text, copy_btn))
            copy_btn.pack(side="left", padx=(0, 6))
        if "regen" in actions:
            try:
                if self._regen_btn is not None:
                    self._regen_btn.destroy()      # chỉ tin cuối mới có nút Hỏi lại
            except Exception:
                pass
            regen = ctk.CTkButton(bar, text="↻ Hỏi lại", width=84, command=self._regenerate_chat, **common)
            regen.pack(side="left")
            self._regen_btn = regen
        box.window_create(mark, window=bar, padx=0, pady=2)
        box.insert(mark, "\n")

    def ask_ai_about(self, message: str, title: str) -> None:
        """Mở cuộc trò chuyện mới ở tab Chat và gửi sẵn câu hỏi (dùng cho nút 'Hỏi AI về câu này')."""
        if self._chat_inflight is not None:
            self.toast.show("AI đang trả lời", "Đợi AI viết xong hoặc bấm “Dừng” rồi thử lại.", PALETTE["warning"], 3500)
            return
        self.tabs.set("Chat với AI")
        conv = self._create_chat()
        conv["title"] = " ".join(title.split())[:40] or "Hỏi AI"
        self._render_chat_view()
        self._refresh_chat_list()
        self._send_chat(message)

    # ----- Hiệu ứng gõ chữ mượt: chữ về theo cụm nhưng được "nhỏ giọt" đều từng ký tự -----
    def _tw_begin(self, tag: str) -> dict[str, Any] | None:
        box = self.chat_view._textbox
        span = box.tag_ranges(tag)
        if len(span) != 2:
            return None
        self.chat_view.configure(state="normal")
        start = str(span[0])
        box.delete(span[0], span[1])
        mark = f"chat_tw_{uuid.uuid4().hex}"
        box.mark_set(mark, start)
        box.mark_gravity(mark, "right")
        box.insert(mark, "AI\n", ("speaker_ai", tag))
        box.insert(mark, "\n\n", (tag,))
        box.mark_set(mark, f"{mark}-2c")      # chữ mới luôn chèn vào trước 2 dòng trống cuối
        self.chat_view.configure(state="disabled")
        state = {"mark": mark, "shown": 0, "target": "", "job": None, "done": False, "final": None}
        self._typewriters[tag] = state
        return state

    def _tw_push(self, tag: str, text: str) -> None:
        state = self._typewriters.get(tag)
        if state is None:
            state = self._tw_begin(tag)
            if state is None:
                return
        state["target"] = text
        if state["job"] is None:
            state["job"] = self.after(16, lambda: self._tw_tick(tag))

    def _tw_tick(self, tag: str) -> None:
        state = self._typewriters.get(tag)
        if state is None:
            return
        state["job"] = None
        box = self.chat_view._textbox
        if len(box.tag_ranges(tag)) != 2:          # khung chat đã được vẽ lại (đổi cuộc trò chuyện...)
            self._typewriters.pop(tag, None)
            final = state.get("final")
            if final:
                final()
            return
        target: str = state["target"]
        backlog = len(target) - state["shown"]
        if backlog > 0:
            # Càng tồn nhiều chữ càng gõ nhanh; ít thì gõ đều ~1-2 ký tự mỗi 16ms.
            step = max(1, int(backlog * (0.30 if state["done"] else 0.07)))
            chunk = target[state["shown"]:state["shown"] + step]
            at_bottom = box.yview()[1] >= 0.98
            self.chat_view.configure(state="normal")
            box.insert(state["mark"], chunk, ("body", tag))
            self.chat_view.configure(state="disabled")
            state["shown"] += step
            if at_bottom:
                box.see("end")
        if state["shown"] < len(target):
            state["job"] = self.after(16, lambda: self._tw_tick(tag))
        elif state["done"]:
            self._typewriters.pop(tag, None)
            try:
                box.mark_unset(state["mark"])
            except Exception:
                pass
            final = state.get("final")
            if final:
                final()

    def _tw_finish(self, tag: str, final: Callable[[], None], failed: bool = False) -> None:
        state = self._typewriters.get(tag)
        if state is None or failed:
            self._tw_cancel(tag)
            final()
            return
        state["done"] = True
        state["final"] = final
        if state["job"] is None:
            state["job"] = self.after(16, lambda: self._tw_tick(tag))

    def _tw_cancel(self, tag: str) -> None:
        state = self._typewriters.pop(tag, None)
        if state and state.get("job") is not None:
            try:
                self.after_cancel(state["job"])
            except Exception:
                pass

    def _clear_chat(self) -> None:
        """Xóa cuộc trò chuyện đang mở (các cuộc khác vẫn được giữ)."""
        if self._chat_active_id is not None:
            self._delete_chat(self._chat_active_id)

    def _on_close(self) -> None:
        self.toast.hide()
        self.listener.stop()
        self.hotkeys.stop()
        self.listen_hotkeys.stop()
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.mouse_trigger.stop()
        if self.recorder:
            self.recorder.stop()
        self._persist_config()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
