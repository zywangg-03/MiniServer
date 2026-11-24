"""Aggregate paired engine runs; report uncertainty rather than a single best run."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from benchmarks.common import write_report


def summarize(report):
    if report.get('kind') != 'serving_evaluation':
        raise ValueError('Expected serving_evaluation report')
    grouped = defaultdict(list)
    for run in report['runs']:
        key = (run['mode'], run['requests'], run['pattern'], run['serving_config']['kv_budget_mb'])
        grouped[key].append(run)
    rows = []
    for (mode, count, pattern, budget), runs in sorted(grouped.items()):
        rates = [r['summary']['output_tokens_per_second'] for r in runs]
        row = {'mode': mode, 'requests': count, 'pattern': pattern, 'kv_budget_mb': budget,
               'repetitions': len(runs), 'all_outputs_match': all(r['summary']['all_outputs_match'] for r in runs),
               'throughput_mean': statistics.mean(rates),
               'throughput_stdev': statistics.stdev(rates) if len(rates) > 1 else None,
               'max_observed_decode_batch': max((max(r['engine']['decode_batch_sizes'], default=0) if r.get('engine') else 0 for r in runs)),
               'peak_allocated_bytes': max((r['cuda_peak_memory']['allocated_bytes'] for r in runs if r['cuda_peak_memory']['allocated_bytes'] is not None), default=None)}
        for metric in ('p95_ttft_seconds', 'p95_queue_seconds', 'p95_latency_seconds', 'p50_tpot_seconds'):
            values = [r['summary'][metric] for r in runs if r['summary'][metric] is not None]
            row[f'mean_run_{metric}'] = statistics.mean(values) if values else None
        rows.append(row)
    for row in rows:
        baseline = next((r for r in rows if r['mode'] == 'hf_sequential' and all(r[k] == row[k] for k in ('requests', 'pattern', 'kv_budget_mb'))), None)
        row['throughput_ratio_to_hf'] = row['throughput_mean'] / baseline['throughput_mean'] if baseline and baseline['throughput_mean'] > 0 else None
    return {'kind': 'serving_comparison', 'passed': report['passed'], 'environment': report['environment'],
            'notes': 'Latency columns average per-run percentiles; not pooled percentiles. HF TTFT/TPOT are unavailable. Ratios compare paired workload settings on this hardware only.', 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/comparison.json'))
    args = parser.parse_args(argv)
    report = summarize(json.loads(args.input.read_text()))
    write_report(args.output, report)
    csv_path = args.output.with_suffix('.csv')
    if report['rows']:
        with csv_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(report['rows'][0]))
            writer.writeheader()
            writer.writerows(report['rows'])
    print(json.dumps({'passed': report['passed'], 'configurations': len(report['rows']),
                      'json': str(args.output), 'csv': str(csv_path)}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
