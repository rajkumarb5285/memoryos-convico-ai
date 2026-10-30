import asyncio
from app.extraction.extractor import DeterministicMemoryExtractor

cases = [
    "I like chai.",
    "I do not like coffee.",
    "Mujhe chai bahut pasand hai.",
    "Mujhe coffee pasand nahi hai.",
    "Mujhe coffee pasand nahin hai.",
    "Mujhe coffee pasand nhi hai.",
    "Mujhe coffee bilkul pasand nahi hai.",
    "Chai pasand hai, coffee nahi.",
    "Mujhe chai pasand hai lekin coffee pasand nahi hai.",
    "I like tea, but I don't like coffee.",
    "Do I like coffee?",
    "If I moved to Delhi, I would look for a job.",
    "My friend likes coffee.",
]

async def run():
    ext = DeterministicMemoryExtractor()
    for c in cases:
        res = await ext.extract(c)
        print(f"INPUT: '{c}'")
        print(f"  is_assertion: {res.is_assertion}")
        print(f"  facts ({len(res.facts)}):")
        for f in res.facts:
            print(f"    - {f.predicate}: {f.object} (polarity={f.polarity}, temporal={f.temporal_status}, subject={f.subject})")
        print()

if __name__ == "__main__":
    asyncio.run(run())
