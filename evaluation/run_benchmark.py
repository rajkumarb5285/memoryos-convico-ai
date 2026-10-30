import asyncio
import time
from datetime import datetime, timezone
import numpy as np

from app.db.session import AsyncSessionLocal
from app.extraction.extractor import DeterministicMemoryExtractor
from app.lifecycle.resolver import lifecycle_resolver
from app.models.character import Character
from app.models.user import User
from app.retrieval.coordinator import retrieval_coordinator
from app.schemas.retrieval import RetrievalRequest


async def run_benchmark(iterations: int = 50, concurrency: int = 5):
    print("=================================================================")
    print(f"Running MEMORYOS Retrieval Benchmark ({iterations} requests, concurrency {concurrency})")
    print("=================================================================\n")

    user_id = f"bench_user_{int(time.time())}"
    char_id = f"bench_char_{int(time.time())}"

    extractor = DeterministicMemoryExtractor()

    # Pre-populate durable store
    async with AsyncSessionLocal() as db:
        user = User(id=user_id, username=user_id, hashed_password="pw")
        db.add(user)
        char = Character(id=char_id, owner_user_id=user_id, name="Benchmark Character")
        db.add(char)
        await db.commit()

        initial_facts = [
            "Main Indore mein rehta hoon.",
            "I recently moved to Pune.",
            "I worked at Google before joining Microsoft.",
            "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai.",
            "My sister studies at Stanford University.",
            "I will travel to Japan next year.",
            "I am now a vegetarian.",
        ]
        for f_text in initial_facts:
            ext = await extractor.extract(f_text)
            if ext.facts:
                await lifecycle_resolver.ingest_facts(
                    db=db,
                    owner_user_id=user_id,
                    character_id=char_id,
                    conversation_id=None,
                    source_message_id=None,
                    facts=ext.facts,
                )
        await db.commit()

    benchmark_queries = [
        "Where do I live now?",
        "Where did I live before?",
        "Which company do I work for?",
        "Where does my sister study?",
        "What do I think about chai?",
        "What is my food preference?",
        "Where do I plan to travel?",
    ]

    latencies_ms = []

    sem = asyncio.Semaphore(concurrency)

    async def worker(query: str):
        async with sem:
            async with AsyncSessionLocal() as db:
                t0 = time.perf_counter()
                await retrieval_coordinator.retrieve(
                    db=db,
                    owner_user_id=user_id,
                    req=RetrievalRequest(query=query, character_id=char_id, limit=5),
                )
                dt = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt)

    # Warm-up (5 requests)
    for q in benchmark_queries[:5]:
        async with AsyncSessionLocal() as db:
            await retrieval_coordinator.retrieve(
                db=db,
                owner_user_id=user_id,
                req=RetrievalRequest(query=q, character_id=char_id, limit=5),
            )

    start_total = time.perf_counter()
    tasks = []
    for i in range(iterations):
        q = benchmark_queries[i % len(benchmark_queries)]
        tasks.append(worker(q))

    await asyncio.gather(*tasks)
    total_time_s = time.perf_counter() - start_total

    avg_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    rps = iterations / total_time_s

    print("-----------------------------------------------------------------")
    print("                     BENCHMARK RESULTS                           ")
    print("-----------------------------------------------------------------")
    print(f"Total Completed Requests:    {iterations}")
    print(f"Concurrency Level:           {concurrency}")
    print(f"Total Duration:              {total_time_s:.2f} seconds")
    print(f"Throughput:                  {rps:.2f} requests/second")
    print(f"Average Latency:             {avg_lat:.2f} ms")
    print(f"P50 (Median) Latency:        {p50_lat:.2f} ms")
    print(f"P95 Latency:                 {p95_lat:.2f} ms")
    print(f"P99 Latency:                 {p99_lat:.2f} ms")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(run_benchmark(iterations=50, concurrency=5))

