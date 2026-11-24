"""Bounded-window server timing summaries, distinct from client-observed timing."""


def latency_summary(records):
    successful = [r for r in records if r['status'] == 'finished']
    result = {'window_requests': len(records), 'successful_requests': len(successful)}
    for key in ('queue_seconds', 'ttft_seconds', 'tpot_seconds', 'latency_seconds'):
        values = sorted(r[key] for r in successful if r.get(key) is not None)
        for percentile in (50, 95):
            index = (len(values) - 1) * percentile / 100
            low = int(index)
            result[f'p{percentile}_{key}'] = (values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)) if values else None
    return result
