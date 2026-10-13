"""A read-only, live mirror of voice sessions, for local observers such as a dashboard widget.

``StreamHub`` fans small JSON messages out to ``/v1/stream`` subscribers.
``ObservedTransport`` wraps a session's real transport and feeds the hub with
what passes through it: assistant audio levels, transcripts and tool calls.
Observers can only listen; nothing here reaches back into a session.

Messages (JSON, every one has ``type``; ``v`` is the protocol version on ``stream.hello``):

- ``stream.hello``  ``{v, active_sessions: [session_id]}``  sent once on connect
- ``session.started`` / ``session.ended``  ``{session_id, transport?}``
- ``audio.level``  ``{session_id, source: "user" | "assistant", level: 0..1}``  ephemeral, ~20 per second
- ``speech``  ``{session_id, active}``  the user started or stopped speaking
- ``transcript``  ``{session_id, role, item_id, text, final}``  ``text`` is cumulative; only ``final`` ones are durable
- ``tool``  ``{session_id, phase: "call" | "result", call_id, name, arguments?, output?}``  full, untruncated
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from array import array
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from speech_to_speech.api.openai_realtime.transports import SessionTransport

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent

logger = logging.getLogger(__name__)

STREAM_VERSION = 1
LEVEL_FRAME_S = 0.05
# Quietest sound that moves the meter, in dBFS. Speech sits roughly -40..-10.
LEVEL_FLOOR_DB = -60.0
SUBSCRIBER_QUEUE_SIZE = 1000
USER_AUDIO_SAMPLE_RATE = 24000

Message = dict[str, Any]


def pcm16_level(pcm: bytes) -> float:
    """Loudness of 16-bit little-endian mono PCM as 0..1 (RMS on a dB scale)."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return 0.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    if mean_square <= 0:
        return 0.0
    db = 10 * math.log10(mean_square / (32768.0**2))
    return round(min(1.0, max(0.0, (db - LEVEL_FLOOR_DB) / -LEVEL_FLOOR_DB)), 3)


def frame_levels(pcm: bytes, sample_rate: int, frame_s: float = LEVEL_FRAME_S) -> list[float]:
    """One level per ``frame_s`` of audio, so a chunk can be replayed at playback speed."""
    frame_bytes = max(2, int(sample_rate * frame_s) * 2)
    return [pcm16_level(pcm[start : start + frame_bytes]) for start in range(0, len(pcm), frame_bytes)]


def is_local_observer(headers: Any, client_host: str | None) -> bool:
    """True only for a non-browser process on this machine.

    Loopback alone is not enough: a Cloudflare tunnel or reverse proxy also
    connects from loopback, and any web page can open a WebSocket to localhost.
    So reject proxied requests (forwarding headers) and anything carrying an
    ``Origin`` (every browser sends one).
    """
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        return False
    if headers.get("origin") is not None:
        return False
    return not any(headers.get(name) for name in ("x-forwarded-for", "forwarded", "cf-connecting-ip", "x-real-ip"))


class Subscription:
    """One observer's bounded queue. A slow observer loses its oldest messages, never stalls a session."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)

    def offer(self, message: Message) -> None:
        if self.queue.full():
            if message["type"] == "audio.level":
                return  # a level is stale before the next one arrives; never evict real content for it
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(message)


class StreamHub:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._subscriptions: set[Subscription] = set()
        self._active: dict[str, str] = {}
        self._transcripts: dict[tuple[str, str], str] = {}
        self._tool_names: dict[tuple[str, str], str] = {}
        self._input_meters: dict[str, tuple[float, float]] = {}
        self._clock = clock

    @property
    def has_observers(self) -> bool:
        return bool(self._subscriptions)

    @property
    def active_sessions(self) -> list[str]:
        return list(self._active)

    def subscribe(self) -> Subscription:
        subscription = Subscription()
        self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        self._subscriptions.discard(subscription)

    def publish(self, message: Message) -> None:
        for subscription in self._subscriptions:
            subscription.offer(message)

    def hello(self) -> Message:
        return {"type": "stream.hello", "v": STREAM_VERSION, "active_sessions": self.active_sessions}

    # ── session lifecycle ──────────────────────────────────────────────

    def session_started(self, session_id: str, transport: str) -> None:
        self._active[session_id] = transport
        self.publish({"type": "session.started", "session_id": session_id, "transport": transport})

    def session_ended(self, session_id: str) -> None:
        if self._active.pop(session_id, None) is None:
            return
        for table in (self._transcripts, self._tool_names):
            for key in [key for key in table if key[0] == session_id]:
                del table[key]
        self._input_meters.pop(session_id, None)
        for source in ("user", "assistant"):
            self.publish_level(session_id, source, 0.0)
        self.publish({"type": "session.ended", "session_id": session_id})

    # ── audio ──────────────────────────────────────────────────────────

    def publish_level(self, session_id: str, source: str, level: float) -> None:
        self.publish({"type": "audio.level", "session_id": session_id, "source": source, "level": level})

    def observe_user_audio(self, session_id: str, pcm: bytes) -> None:
        """Mic audio arrives in real time, so publish at most one (peak) level per frame."""
        if not self.has_observers:
            return
        last_emit, peak = self._input_meters.get(session_id, (0.0, 0.0))
        peak = max(peak, pcm16_level(pcm))
        now = self._clock()
        if now - last_emit < LEVEL_FRAME_S:
            self._input_meters[session_id] = (last_emit, peak)
            return
        self._input_meters[session_id] = (now, 0.0)
        self.publish_level(session_id, "user", peak)

    # ── what the session said and did ──────────────────────────────────

    def observe_server_event(self, session_id: str, event: Any) -> None:
        """Turn one outgoing Realtime event into stream messages (most events produce none)."""
        kind = getattr(event, "type", "")
        item_id = getattr(event, "item_id", "")
        if kind == "conversation.item.input_audio_transcription.delta":
            self._partial(session_id, "user", item_id, event.delta)
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._final(session_id, "user", item_id, event.transcript)
        elif kind == "conversation.item.created":
            # Messages the client puts into the conversation: typed ones, and the remembered history a new session
            # replays (both roles, or the replayed conversation would show only what the user said). Spoken user
            # messages and the assistant's own replies arrive through the transcription/response events instead.
            item = event.item
            role = getattr(item, "role", "")
            if role in ("user", "assistant"):
                text = "".join(getattr(part, "text", None) or "" for part in getattr(item, "content", None) or [])
                if text.strip() and getattr(item, "id", None):
                    self._final(session_id, role, item.id, text)
        elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
            self._partial(session_id, "assistant", item_id, event.delta)
        elif kind == "response.output_audio_transcript.done":
            self._final(session_id, "assistant", item_id, event.transcript)
        elif kind == "response.output_text.done":
            self._final(session_id, "assistant", item_id, event.text)
        elif kind == "response.function_call_arguments.done":
            self._tool_names[(session_id, event.call_id)] = event.name
            self.publish(
                {
                    "type": "tool",
                    "session_id": session_id,
                    "phase": "call",
                    "call_id": event.call_id,
                    "name": event.name,
                    "arguments": event.arguments,
                }
            )
        elif kind in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
            self.publish({"type": "speech", "session_id": session_id, "active": kind.endswith("started")})

    def observe_tool_result(self, session_id: str, call_id: str, output: str) -> None:
        name = self._tool_names.get((session_id, call_id), "")
        self.publish(
            {
                "type": "tool",
                "session_id": session_id,
                "phase": "result",
                "call_id": call_id,
                "name": name,
                "output": output,
            }
        )

    def _partial(self, session_id: str, role: str, item_id: str, delta: str) -> None:
        key = (session_id, item_id)
        text = self._transcripts.get(key, "") + delta
        self._transcripts[key] = text
        self._transcript(session_id, role, item_id, text, final=False)

    def _final(self, session_id: str, role: str, item_id: str, text: str) -> None:
        self._transcripts.pop((session_id, item_id), None)
        self._transcript(session_id, role, item_id, text, final=True)

    def _transcript(self, session_id: str, role: str, item_id: str, text: str, *, final: bool) -> None:
        self.publish(
            {
                "type": "transcript",
                "session_id": session_id,
                "role": role,
                "item_id": item_id,
                "text": text,
                "final": final,
            }
        )


class ObservedTransport(SessionTransport):
    """Passes everything to *inner* unchanged while feeding the hub."""

    def __init__(self, inner: SessionTransport, hub: StreamHub) -> None:
        self.inner = inner
        self.hub = hub
        self.session_id = ""
        self._playout_end = 0.0
        self._level_timers: set[asyncio.TimerHandle] = set()

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.inner.kind

    def start(self, session_id: str) -> None:
        self.session_id = session_id
        self.hub.session_started(session_id, self.kind)

    def finish(self) -> None:
        self._cancel_level_timers()
        if self.session_id:
            self.hub.session_ended(self.session_id)

    async def send_events(self, events: list[ServerEvent]) -> None:
        if self.session_id:
            for event in events:
                self._observe(event)
        await self.inner.send_events(events)

    async def send_audio_chunk(
        self, service: RealtimeService, session_id: str, pcm: bytes, response_key: str | None = None
    ) -> None:
        self._schedule_levels(pcm)
        await self.inner.send_audio_chunk(service, session_id, pcm, response_key)

    def discard_pending_audio(self) -> None:
        self._cancel_level_timers()
        if self.session_id:
            self.hub.publish_level(self.session_id, "assistant", 0.0)
        self.inner.discard_pending_audio()

    async def close(self) -> None:
        await self.inner.close()

    def observe_user_audio(self, pcm: bytes) -> None:
        if self.session_id:
            self.hub.observe_user_audio(self.session_id, pcm)

    def observe_tool_result(self, call_id: str, output: str) -> None:
        if self.session_id:
            self.hub.observe_tool_result(self.session_id, call_id, output)

    def _observe(self, event: ServerEvent) -> None:
        try:
            self.hub.observe_server_event(self.session_id, event)
        except Exception:  # noqa: BLE001 - observing must never break a live session
            logger.exception("Could not mirror %s to stream observers", getattr(event, "type", "event"))

    def _schedule_levels(self, pcm: bytes) -> None:
        """TTS audio is produced faster than it plays; replay its levels on the playback clock."""
        if not self.session_id or not self.hub.has_observers:
            return
        from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE

        loop = asyncio.get_running_loop()
        start = max(loop.time(), self._playout_end)
        levels = frame_levels(pcm, PIPELINE_SAMPLE_RATE)
        for index, level in enumerate(levels):
            self._at(loop, start + index * LEVEL_FRAME_S, level)
        self._playout_end = start + len(levels) * LEVEL_FRAME_S
        self._at(loop, self._playout_end, 0.0, settle=True)  # back to silence, unless more audio follows

    def _at(self, loop: asyncio.AbstractEventLoop, when: float, level: float, *, settle: bool = False) -> None:
        handle: asyncio.TimerHandle

        def fire() -> None:
            self._level_timers.discard(handle)
            if settle and loop.time() < self._playout_end - 1e-3:
                return  # more audio was queued behind this marker
            self.hub.publish_level(self.session_id, "assistant", level)

        handle = loop.call_at(when, fire)
        self._level_timers.add(handle)

    def _cancel_level_timers(self) -> None:
        for handle in self._level_timers:
            handle.cancel()
        self._level_timers.clear()
        self._playout_end = 0.0


async def serve_observer(ws: WebSocket, hub: StreamHub, stop: Any) -> None:
    """Forward hub messages to one accepted observer until either side goes away."""
    subscription = hub.subscribe()
    try:
        await ws.send_json(hub.hello())

        async def pump() -> None:
            while not stop.is_set():
                try:
                    message = await asyncio.wait_for(subscription.queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await ws.send_json(message)

        async def drain_incoming() -> None:
            # Observers are read-only: whatever they send is ignored. Receiving just notices the disconnect.
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(pump()), asyncio.create_task(drain_incoming())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            task.exception()  # a disconnect surfaces here; there is nothing to recover
    finally:
        hub.unsubscribe(subscription)
