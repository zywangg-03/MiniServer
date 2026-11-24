"""HTTP SSE latency measurements. Every attempted request is retained in JSON."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import httpx

from benchmarks.common import DEFAULT_PROMPTS, percentile, positive_int, write_report


async def one_request(client, index, prompt, max_new_tokens, delay=0):
    await asyncio.sleep(delay)
    started = time.perf_counter()
    token_times, token_ids = [], []
    record = {'request_index': index, 'prompt': prompt, 'status': 'error', 'error': None}
    try:
        async with client.stream('POST', '/v1/generate/stream', json={'prompt': prompt, 'max_new_tokens': max_new_tokens, 'temperature': 0}) as response:
            record['http_status'] = response.status_code
            if response.status_code != 200:
                record['error'] = (await response.aread()).decode(errors='replace')
            else:
                terminal = None
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):
                        continue
                    event = json.loads(line[6:])
                    if event['type'] == 'token':
                        token_times.append(time.perf_counter())
                        token_ids.append(event['token_id'])
                    elif event['type'] in ('done', 'error'):
                        terminal = event
                        break
                if terminal is None:
                    record['error'] = 'Stream ended without a terminal event'
                else:
                    record['server_result'] = terminal
                    record['status'] = terminal['status']
                    record['error'] = terminal.get('error')
                    if terminal.get('output_ids') != token_ids:
                        record['status'] = 'error'
                        record['error'] = 'Stream token IDs differ from terminal result'
    except Exception as error:
        record['error'] = str(error)
    record.update({
        'output_ids': token_ids,
        'token_arrival_offsets_seconds': [t - started for t in token_times],
        'ttft_seconds': token_times[0] - started if token_times else None,
        'tpot_seconds': (token_times[-1] - token_times[0]) / (len(token_times) - 1) if len(token_times) > 1 else None,
        'latency_seconds': time.perf_counter() - started,
    })
    return record


async def run(args):
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.url.rstrip('/'), timeout=timeout,
                                 limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency), trust_env=False) as client:
        health = await client.get('/health')
        health.raise_for_status()
        if not health.json().get('ready'):
            raise RuntimeError('Server is not ready')
        for _ in range(args.warmup):
            r = await one_request(client, 0, args.prompt or DEFAULT_PROMPTS[0], args.max_new_tokens)
            if r['status'] != 'finished':
                raise RuntimeError(f'Warmup failed: {r}')
        before = (await client.get('/metrics')).json()
        started = time.perf_counter()
        runs = []
        for repetition in range(args.repetitions):
            records = await asyncio.gather(*[
                one_request(client, i, args.prompt or DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)], args.max_new_tokens,
                            i * args.arrival_interval if args.pattern == 'staggered' else 0)
                for i in range(args.concurrency)
            ])
            for r in records:
                r['repetition'] = repetition
            runs.extend(records)
        elapsed = time.perf_counter() - started
        after = (await client.get('/metrics')).json()
    successes = [r for r in runs if r['status'] == 'finished']
    summary = {'attempted': len(runs), 'completed': len(successes), 'failed': len(runs) - len(successes),
               'wall_seconds': elapsed, 'output_tokens_per_second': sum(len(r['output_ids']) for r in successes) / elapsed}
    for metric in ('ttft_seconds', 'tpot_seconds', 'latency_seconds'):
        values = [r[metric] for r in successes if r[metric] is not None]
        for p in (50, 95):
            summary[f'p{p}_{metric}'] = percentile(values, p / 100) if values else None
    return {'schema_version': 1, 'kind': 'http_load_test', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Client-observed SSE timing; token times include transport and buffering; successful-request latency percentiles, all attempts retained.',
            'workload': {k: v for k, v in vars(args).items() if k != 'output'},
            'passed': len(successes) == len(runs), 'summary': summary,
            'metrics_before': before, 'metrics_after': after, 'records': runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--concurrency', type=positive_int, default=8)
    parser.add_argument('--max-new-tokens', type=positive_int, default=64)
    parser.add_argument('--repetitions', type=positive_int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--pattern', choices=['burst', 'staggered'], default='burst')
    parser.add_argument('--arrival-interval', type=float, default=0.05)
    parser.add_argument('--prompt')
    parser.add_argument('--output', type=Path, default=Path('results/http_load.json'))
    args = parser.parse_args(argv)
    if args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.arrival_interval) or args.arrival_interval < 0:
        parser.error('Invalid warmup, timeout or arrival interval')
    report = asyncio.run(run(args))
    write_report(args.output, report)
    print(json.dumps(report['summary'], indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
