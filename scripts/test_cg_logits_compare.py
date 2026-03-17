"""
Minimal test: send identical requests to sglang server and compare outputs.
Uses two servers: one with CG, one without CG.
If outputs differ on compressed requests, the bug is in sglang's CG integration.
"""
import asyncio
import httpx
import json
import sys

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

async def send_request(port, messages, temperature=0.0, max_tokens=50):
    """Send a request and return the output text."""
    client = httpx.AsyncClient(timeout=60.0)
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = await client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload)
    data = resp.json()
    await client.aclose()
    return data["choices"][0]["message"]["content"]


async def main():
    cg_port = int(sys.argv[1])     # CG server port
    nocg_port = int(sys.argv[2])   # non-CG server port
    
    # Test 1: Simple short request (no compression)
    messages_short = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": "What is 2+2? Answer briefly."},
    ]
    
    print("=== Test 1: Short request (no compression) ===")
    for trial in range(3):
        out_cg = await send_request(cg_port, messages_short, temperature=0.0, max_tokens=20)
        out_nocg = await send_request(nocg_port, messages_short, temperature=0.0, max_tokens=20)
        match = out_cg == out_nocg
        print(f"  trial {trial}: match={match}")
        if not match:
            print(f"    CG:   {out_cg[:100]}")
            print(f"    NoCG: {out_nocg[:100]}")
    
    # Test 2: Long request that triggers compression, then continuation
    question = "Let P(x) be a monic polynomial of degree 3. Suppose that P(x) has remainder R(x) when it is divided by (x - 1)(x - 4), and remainder 2R(x) when it is divided by (x - 2)(x - 3). Given that P(0) = 5, find P(5)."
    
    messages_init = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": f"Please solve step by step. Put answer in \\boxed{{}}. Question: {question}"},
    ]
    
    print("\n=== Test 2: Initial draft (may not trigger compression) ===")
    out_cg_draft = await send_request(cg_port, messages_init, temperature=0.0, max_tokens=500)
    out_nocg_draft = await send_request(nocg_port, messages_init, temperature=0.0, max_tokens=500)
    print(f"  CG  length: {len(out_cg_draft)}")
    print(f"  NoCG length: {len(out_nocg_draft)}")
    print(f"  Match: {out_cg_draft == out_nocg_draft}")
    if out_cg_draft != out_nocg_draft:
        # Find first difference
        for i in range(min(len(out_cg_draft), len(out_nocg_draft))):
            if out_cg_draft[i] != out_nocg_draft[i]:
                print(f"  First diff at char {i}:")
                print(f"    CG:   ...{out_cg_draft[max(0,i-20):i+20]}...")
                print(f"    NoCG: ...{out_nocg_draft[max(0,i-20):i+20]}...")
                break
    
    # Test 3: Continuation (should trigger compression on CG server)
    messages_cont = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": f"Please solve step by step. Put answer in \\boxed{{}}. Question: {question}"},
        {"role": "assistant", "content": out_nocg_draft},  # use nocg output as shared history
    ]
    
    print("\n=== Test 3: Continuation (triggers compression, temp=0.0) ===")
    out_cg_cont = await send_request(cg_port, messages_cont, temperature=0.0, max_tokens=500)
    out_nocg_cont = await send_request(nocg_port, messages_cont, temperature=0.0, max_tokens=500)
    print(f"  CG  length: {len(out_cg_cont)}")
    print(f"  NoCG length: {len(out_nocg_cont)}")
    print(f"  Match: {out_cg_cont == out_nocg_cont}")
    if out_cg_cont != out_nocg_cont:
        for i in range(min(len(out_cg_cont), len(out_nocg_cont))):
            if out_cg_cont[i] != out_nocg_cont[i]:
                print(f"  First diff at char {i}:")
                print(f"    CG:   ...{out_cg_cont[max(0,i-20):i+20]}...")
                print(f"    NoCG: ...{out_nocg_cont[max(0,i-20):i+20]}...")
                break
        # Check for repetition
        for label, text in [("CG", out_cg_cont), ("NoCG", out_nocg_cont)]:
            words = text.split()
            if len(words) > 20:
                chunk = " ".join(words[-10:])
                rep = text.count(chunk[:30])
                if rep > 2:
                    print(f"  {label}: REPETITION detected (count={rep})")
    
    print("\n=== Test 4: Multiple continuations (deeper compression) ===")
    combined = out_nocg_draft + "\n\n" + out_nocg_cont
    messages_deep = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": f"Please solve. Question: {question}"},
        {"role": "assistant", "content": combined},
    ]
    out_cg_deep = await send_request(cg_port, messages_deep, temperature=0.0, max_tokens=500)
    out_nocg_deep = await send_request(nocg_port, messages_deep, temperature=0.0, max_tokens=500)
    print(f"  CG  length: {len(out_cg_deep)}")
    print(f"  NoCG length: {len(out_nocg_deep)}")
    print(f"  Match: {out_cg_deep == out_nocg_deep}")
    if out_cg_deep != out_nocg_deep:
        for i in range(min(len(out_cg_deep), len(out_nocg_deep))):
            if out_cg_deep[i] != out_nocg_deep[i]:
                print(f"  First diff at char {i}:")
                print(f"    CG:   ...{out_cg_deep[max(0,i-20):i+20]}...")
                print(f"    NoCG: ...{out_nocg_deep[max(0,i-20):i+20]}...")
                break

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cg_port> <nocg_port>")
        sys.exit(1)
    asyncio.run(main())
