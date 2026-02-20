"""
Example client demonstrating all features of the streaming pipeline.
"""
import asyncio
import json
import base64
import wave
import websockets
from datetime import datetime
from pathlib import Path


class StreamingAvatarClient:
    """
    Complete example client for real-time conversational avatar.
    
    Features demonstrated:
    - Session creation
    - Streaming text, audio, and blendshapes
    - Interrupt handling
    - Buffer adjustment
    - Audio playback
    - Blendshape visualization
    - Performance monitoring
    """
    
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.session_id = None
        self.audio_chunks = []
        self.blendshape_frames = []
        self.text_chunks = []
    
    async def create_session(self, url: str = None, files: list = None):
        """Create a RAG session from URL or files."""
        import aiohttp
        
        print(f"\n{'='*60}")
        print("Creating Session")
        print(f"{'='*60}\n")
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            
            if url:
                data.add_field('url', url)
                print(f"Processing URL: {url}")
            
            if files:
                for file_path in files:
                    data.add_field(
                        'files',
                        open(file_path, 'rb'),
                        filename=Path(file_path).name
                    )
                print(f"Processing files: {files}")
            
            async with session.post(
                f"{self.base_url}/process",
                data=data
            ) as response:
                result = await response.json()
                self.session_id = result['session_id']
                print(f"\n✅ Session created: {self.session_id}")
                print(f"   Message: {result['message']}")
                return self.session_id
    
    async def stream_conversation(
        self,
        question: str,
        use_qwen: bool = True,
        use_optimized_bs: bool = True,
        save_audio: bool = True,
        save_blendshapes: bool = True,
    ):
        """Stream a conversation with full pipeline."""
        if not self.session_id:
            raise ValueError("No session created. Call create_session() first.")
        
        print(f"\n{'='*60}")
        print("Starting Streaming Conversation")
        print(f"{'='*60}\n")
        print(f"Question: {question}")
        print(f"Using Qwen TTS: {use_qwen}")
        print(f"Using Optimized BS: {use_optimized_bs}\n")
        
        uri = f"{self.ws_url}/ws/infer/kyutai"
        
        async with websockets.connect(uri) as ws:
            # Send start message
            start_msg = {
                "type": "start",
                "session_id": self.session_id,
                "question": question,
                "use_qwen": use_qwen,
                "use_optimized_bs": use_optimized_bs,
                "return_audio": True,
            }
            
            await ws.send(json.dumps(start_msg))
            print(f"[{datetime.now()}] Sent start message\n")
            
            # Reset storage
            self.audio_chunks = []
            self.blendshape_frames = []
            self.text_chunks = []
            
            # Receive messages
            try:
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    await self._handle_message(data)
                    
                    # Check for completion
                    if data.get("type") == "status":
                        status = data.get("status")
                        if status in ["complete", "error", "interrupted"]:
                            break
            
            except websockets.exceptions.ConnectionClosed:
                print(f"\n[{datetime.now()}] Connection closed")
        
        # Save outputs
        if save_audio and self.audio_chunks:
            self._save_audio("output_audio.wav")
        
        if save_blendshapes and self.blendshape_frames:
            self._save_blendshapes("output_blendshapes.json")
        
        print(f"\n{'='*60}")
        print("Conversation Complete")
        print(f"{'='*60}\n")
        print(f"Text chunks received: {len(self.text_chunks)}")
        print(f"Audio chunks received: {len(self.audio_chunks)}")
        print(f"Blendshape frames received: {len(self.blendshape_frames)}")
    
    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")
        
        if msg_type == "text_chunk":
            self.text_chunks.append(data["text"])
            print(f"📝 [{data['sentence_index']}] {data['text']}")
        
        elif msg_type == "audio_chunk":
            if not data.get("is_final"):
                audio_b64 = data.get("audio_base64", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    self.audio_chunks.append({
                        "bytes": audio_bytes,
                        "start_time": data["start_time"],
                        "end_time": data["end_time"],
                        "sample_rate": data["sample_rate"],
                    })
                    duration = data["end_time"] - data["start_time"]
                    print(f"🔊 Audio chunk {data['chunk_index']}: {duration:.2f}s")
        
        elif msg_type == "blendshapes":
            if not data.get("is_final"):
                frames = data.get("frames", [])
                self.blendshape_frames.extend(frames)
                print(f"😊 Blendshapes {data['chunk_index']}: {len(frames)} frames")
        
        elif msg_type == "idle_frames":
            frames = data.get("frames", [])
            self.blendshape_frames.extend(frames)
            print(f"😐 Idle frames: {len(frames)} frames")
        
        elif msg_type == "status":
            status = data["status"]
            message = data["message"]
            
            emoji = {
                "processing": "⚙️",
                "complete": "✅",
                "error": "❌",
                "interrupted": "⏸️",
                "warning": "⚠️",
            }.get(status, "ℹ️")
            
            print(f"{emoji} Status: {status} - {message}")
    
    def _save_audio(self, filename: str):
        """Save audio chunks to WAV file."""
        if not self.audio_chunks:
            return
        
        print(f"\n💾 Saving audio to {filename}...")
        
        # Combine all audio chunks
        sample_rate = self.audio_chunks[0]["sample_rate"]
        combined_audio = b"".join(chunk["bytes"] for chunk in self.audio_chunks)
        
        # Write WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(combined_audio)
        
        print(f"✅ Audio saved: {filename}")
    
    def _save_blendshapes(self, filename: str):
        """Save blendshape frames to JSON file."""
        if not self.blendshape_frames:
            return
        
        print(f"💾 Saving blendshapes to {filename}...")
        
        with open(filename, 'w') as f:
            json.dump({
                "frame_rate": 60,
                "total_frames": len(self.blendshape_frames),
                "frames": self.blendshape_frames,
            }, f, indent=2)
        
        print(f"✅ Blendshapes saved: {filename}")
    
    async def demonstrate_interrupt(self, question: str, interrupt_after: float = 2.0):
        """Demonstrate interrupt functionality."""
        if not self.session_id:
            raise ValueError("No session created. Call create_session() first.")
        
        print(f"\n{'='*60}")
        print("Demonstrating Interrupt")
        print(f"{'='*60}\n")
        print(f"Will interrupt after {interrupt_after}s\n")
        
        uri = f"{self.ws_url}/ws/infer/kyutai"
        
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "session_id": self.session_id,
                "question": question,
                "use_qwen": True,
                "return_audio": False,
            }))
            
            start_time = asyncio.get_event_loop().time()
            interrupted = False
            
            try:
                while True:
                    # Check if we should interrupt
                    if not interrupted and (asyncio.get_event_loop().time() - start_time) > interrupt_after:
                        print(f"\n⏸️  Sending interrupt...\n")
                        await ws.send(json.dumps({"type": "interrupt"}))
                        interrupted = True
                    
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "text_chunk":
                        print(f"📝 {data['text'][:50]}...")
                    elif data.get("type") == "status":
                        print(f"ℹ️  {data['status']}: {data['message']}")
                        if data['status'] in ["complete", "interrupted", "error"]:
                            break
            
            except asyncio.TimeoutError:
                pass
        
        print(f"\n✅ Interrupt demonstration complete")
    
    async def demonstrate_buffer_adjustment(self, question: str):
        """Demonstrate adaptive buffer adjustment."""
        if not self.session_id:
            raise ValueError("No session created. Call create_session() first.")
        
        print(f"\n{'='*60}")
        print("Demonstrating Buffer Adjustment")
        print(f"{'='*60}\n")
        
        uri = f"{self.ws_url}/ws/infer/kyutai"
        
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "session_id": self.session_id,
                "question": question,
                "use_qwen": True,
                "return_audio": False,
            }))
            
            # Adjust buffer after 1 second
            await asyncio.sleep(1)
            print("\n📊 Adjusting buffer size to 5 (smoother playback)...\n")
            await ws.send(json.dumps({
                "type": "buffer_adjust",
                "target_size": 5
            }))
            
            # Receive messages
            try:
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "status":
                        print(f"ℹ️  {data['status']}: {data['message']}")
                        if data['status'] in ["complete", "error"]:
                            break
            
            except websockets.exceptions.ConnectionClosed:
                pass
        
        print(f"\n✅ Buffer adjustment demonstration complete")
    
    async def get_performance_metrics(self):
        """Get performance metrics from server."""
        import aiohttp
        
        print(f"\n{'='*60}")
        print("Performance Metrics")
        print(f"{'='*60}\n")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/performance/summary") as response:
                metrics = await response.json()
                
                print(f"Session Duration: {metrics['session_duration']}s")
                print(f"Total Sentences: {metrics['total_sentences']}")
                print(f"Total Frames: {metrics['total_frames']}")
                print(f"Realtime Factor: {metrics['realtime_factor']}x")
                print(f"\nEnd-to-End Latency:")
                print(f"  Average: {metrics['e2e_latency']['avg']}s")
                print(f"  Min: {metrics['e2e_latency']['min']}s")
                print(f"  Max: {metrics['e2e_latency']['max']}s")
                print(f"\nBuffer Health:")
                print(f"  Average: {metrics['buffer_health']['avg']}")
                print(f"\nStage Performance:")
                for stage, stats in metrics['stages'].items():
                    print(f"  {stage}:")
                    print(f"    Avg Time: {stats['avg_time']}s")
                    print(f"    Error Rate: {stats['error_rate']}%")


async def main():
    """Run example demonstrations."""
    client = StreamingAvatarClient()
    
    # Demo 1: Create session
    await client.create_session(
        url="https://en.wikipedia.org/wiki/Artificial_intelligence"
    )
    
    # Demo 2: Basic streaming
    await client.stream_conversation(
        question="What is artificial intelligence in simple terms?",
        use_qwen=True,
        use_optimized_bs=True,
        save_audio=True,
        save_blendshapes=True,
    )
    
    # Demo 3: Interrupt
    await client.demonstrate_interrupt(
        question="Tell me a very long story about space exploration",
        interrupt_after=2.0
    )
    
    # Demo 4: Buffer adjustment
    await client.demonstrate_buffer_adjustment(
        question="Explain quantum computing"
    )
    
    # Demo 5: Performance metrics
    await client.get_performance_metrics()
    
    print(f"\n{'='*60}")
    print("All Demonstrations Complete!")
    print(f"{'='*60}\n")
    print("Check the following files:")
    print("  - output_audio.wav")
    print("  - output_blendshapes.json")


if __name__ == "__main__":
    asyncio.run(main())
