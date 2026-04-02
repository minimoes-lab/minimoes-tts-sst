import asyncio
import base64
import json
import os
import struct
import threading
from typing import Optional

import numpy as np
import requests
import websockets

try:
    import winsound  # type: ignore
except Exception:
    winsound = None


def _write_wav_pcm16(path: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm)
    riff_size = 36 + data_size

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


class _PCM16RingBuffer:
    def __init__(self, capacity_samples: int):
        self._buf = np.zeros((capacity_samples,), dtype=np.int16)
        self._cap = int(capacity_samples)
        self._w = 0
        self._r = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        x = np.asarray(samples, dtype=np.int16).reshape(-1)
        n = int(x.size)
        if n >= self._cap:
            x = x[-self._cap :]
            n = int(x.size)
        with self._lock:
            end = self._w + n
            if end <= self._cap:
                self._buf[self._w : end] = x
            else:
                first = self._cap - self._w
                self._buf[self._w :] = x[:first]
                self._buf[: end % self._cap] = x[first:]
            self._w = end % self._cap
            if self._w == self._r:
                self._r = (self._r + n) % self._cap

    def read(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.int16)
        out = np.zeros((n,), dtype=np.int16)
        with self._lock:
            avail = (self._w - self._r) % self._cap
            take = min(n, avail)
            if take == 0:
                return out
            end = self._r + take
            if end <= self._cap:
                out[:take] = self._buf[self._r : end]
            else:
                first = self._cap - self._r
                out[:first] = self._buf[self._r :]
                out[first:take] = self._buf[: end % self._cap]
            self._r = end % self._cap
        return out


async def main():
    pod_url = os.environ.get("POD_URL", "http://127.0.0.1:7860").rstrip("/")
    test_url = os.environ.get("TEST_URL", "http://httpbin.org/html")
    question = os.environ.get("QUESTION", "Hello how are you?")
    voice_preset = os.environ.get("VOICE_PRESET", "").strip() or None
    tts_instruct = os.environ.get("TTS_INSTRUCT", "").strip() or None
    voice_id = os.environ.get("VOICE_ID", "default").strip() or "default"
    chunk_ms = int(os.environ.get("CHUNK_MS", "50"))
    play_live = os.environ.get("PLAY_LIVE", "1") not in ("0", "false", "False")
    play_file = os.environ.get("PLAY_FILE", "0") in ("1", "true", "True")
    jitter_chunks = int(os.environ.get("JITTER_CHUNKS", "1"))
    list_devices = os.environ.get("LIST_DEVICES", "0") in ("1", "true", "True")
    output_device = os.environ.get("OUTPUT_DEVICE")

    out_root = os.path.join(os.path.expanduser("~"), "Desktop", "ws_streaming_out")
    os.makedirs(out_root, exist_ok=True)
    bs_out = os.path.join(out_root, "blendshapes")
    os.makedirs(bs_out, exist_ok=True)

    # 1) /process
    req = {"question": question, "return_audio": True, "return_csv": False}
    resp = requests.post(
        f"{pod_url}/process",
        files={"request_raw": ("request_raw.json", json.dumps(req), "application/json")},
        data={"url": test_url},
        timeout=120,
    )
    resp.raise_for_status()
    session_id = resp.json().get("session_id")
    if not session_id:
        raise RuntimeError("/process returned empty session_id")

    # 2) WS
    ws_base = pod_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_base}/ws/infer/kyutai"

    pcm_chunks: dict[int, bytes] = {}
    sample_rate = 24000
    channels = 1

    ring: Optional[_PCM16RingBuffer] = None
    play_ready = False
    expected_idx: Optional[int] = None

    sd = None
    stream = None
    sd_device = None
    if play_live:
        try:
            import sounddevice as _sd  # type: ignore

            sd = _sd
            if list_devices:
                try:
                    print("=== sounddevice devices ===")
                    print(sd.query_devices())
                except Exception as e:
                    print(f"WARN Failed to query devices: {e}")

            if output_device is not None and str(output_device).strip() != "":
                od = str(output_device).strip()
                sd_device = int(od) if od.isdigit() else od
        except Exception:
            sd = None

    got_final_audio = False
    got_final_bs = False

    try:
        async with websockets.connect(
            ws_url,
            max_size=None,
            ping_interval=None,
            close_timeout=30,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "start",
                        "session_id": session_id,
                        "question": question,
                        "return_audio": True,
                        "chunk_ms": chunk_ms,
                        "voice_preset": voice_preset,
                        "voice_id": voice_id,
                        "tts_instruct": tts_instruct,
                    }
                )
            )

            while True:
                raw = await ws.recv()
                msg = json.loads(raw)

                mtype = msg.get("type")
                if mtype == "status":
                    status = msg.get("status")
                    message = msg.get("message")
                    print(f"[server status] {status}: {message}")
                    if status == "error":
                        raise RuntimeError(message or "Streaming error")
                elif mtype == "audio_chunk":
                    if msg.get("is_final"):
                        got_final_audio = True
                    sample_rate = int(msg.get("sample_rate") or sample_rate)
                    channels = int(msg.get("channels") or channels)

                    if ring is None:
                        cap = max(1, int(sample_rate * 10))
                        ring = _PCM16RingBuffer(capacity_samples=cap)

                        if sd is not None and channels == 1:
                            def _cb(outdata, frames, time_info, status):
                                chunk = ring.read(frames)
                                outdata[:, 0] = (chunk.astype(np.float32) / 32768.0)

                            stream = sd.OutputStream(
                                samplerate=sample_rate,
                                channels=1,
                                dtype="float32",
                                callback=_cb,
                                blocksize=0,
                                device=sd_device,
                            )
                            stream.start()

                    b64 = msg.get("audio_bytes_base64") or ""
                    if b64:
                        idx = msg.get("chunk_index")
                        if isinstance(idx, int):
                            pcm_chunks[idx] = base64.b64decode(b64)

                            if expected_idx is None:
                                expected_idx = idx

                            if ring is not None and expected_idx is not None:
                                while expected_idx in pcm_chunks:
                                    if not play_ready:
                                        contiguous = 0
                                        k = expected_idx
                                        while k in pcm_chunks and contiguous < max(1, jitter_chunks):
                                            contiguous += 1
                                            k += 1
                                        if contiguous >= max(1, jitter_chunks):
                                            play_ready = True
                                    if not play_ready:
                                        break

                                    raw_bytes = pcm_chunks[expected_idx]
                                    samples = np.frombuffer(raw_bytes, dtype=np.int16)
                                    ring.write(samples)
                                    expected_idx += 1
                elif mtype == "blendshapes":
                    si = msg.get("sentence_index")
                    ci = msg.get("chunk_index")
                    if isinstance(si, int) and isinstance(ci, int):
                        out_path = os.path.join(bs_out, f"blendshapes_s{si}_c{ci}.json")
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(msg, f, ensure_ascii=False)
                    if msg.get("is_final"):
                        got_final_bs = True

                if got_final_audio and got_final_bs:
                    break
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket closed by server (OK).")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket closed by server with error: {e}")

    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    ordered = b"".join(pcm_chunks[i] for i in sorted(pcm_chunks.keys()))
    if ordered:
        out_path = os.path.join(out_root, "final_pcm_stream.wav")
        _write_wav_pcm16(out_path, ordered, sample_rate, channels)
        print(
            f"Wrote: {out_path} bytes={len(ordered)} sr={sample_rate} ch={channels}"
        )
        if play_file and winsound is not None:
            try:
                winsound.PlaySound(out_path, winsound.SND_FILENAME)
            except Exception as e:
                print(f"WARN Failed to play file: {e}")
    else:
        print("No PCM audio received; nothing to write.")

    if play_live and sd is None:
        print("Live playback requested but sounddevice is not installed. Install with: pip install sounddevice")


if __name__ == "__main__":
    asyncio.run(main())
