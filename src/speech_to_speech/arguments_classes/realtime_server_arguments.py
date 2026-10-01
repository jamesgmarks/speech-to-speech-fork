from dataclasses import dataclass, field


@dataclass
class RealtimeServerArguments:
    conversation_store_dir: str = field(
        default="~/.speech-to-speech/conversations",
        metadata={"help": "Directory for durable conversation context requested by reconnecting clients."},
    )
    host: str = field(
        default="127.0.0.1",
        metadata={
            "help": "Host interface for the Realtime server. Default is 127.0.0.1. Pass 0.0.0.0 explicitly "
            "to expose the unauthenticated API on the network."
        },
    )
    port: int = field(
        default=8765,
        metadata={"help": "Port for the Realtime HTTP/WebSocket server. Default is 8765."},
    )


@dataclass
class LocalRealtimeServerArguments:
    port: int = field(
        default=8765,
        metadata={"help": "Loopback port for the local Realtime server and audio client. Default is 8765."},
    )
