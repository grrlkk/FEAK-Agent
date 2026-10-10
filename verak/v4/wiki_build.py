"""Low-priority, resumable local Wikipedia preparation; no model API calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from urllib.request import Request, urlopen

from .wiki_local import EXTRACTOR_VERSION, LICENSE, LICENSE_URL, TOKENIZER_VERSION

ROOT = Path('/home/chanwoo/FEAK-Agent/verak/v4/outputs/scale2/search')
USER_AGENT = 'FEAK-LocalResearch/1.0 (local Korean Wikipedia index; no live search requests)'


def constrain():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    chosen = set(os.sched_getaffinity(0)) & set(range(116, 120))
    os.sched_setaffinity(0, chosen or sorted(os.sched_getaffinity(0))[-2:])
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen artifact changed: {path}')
    if not path.exists():
        write(path, value)
    return value


def status(root, stage, **values):
    value = {'stage': stage, 'at': datetime.now(timezone.utc).isoformat(),
             'gpu_used': False, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'nice': os.nice(0), **values}
    write(root / 'build_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fetch(url):
    # Only public Wikimedia dump metadata; no user text or query is accepted here.
    if not url.startswith('https://dumps.wikimedia.org/kowiki/'):
        raise ValueError('Only the official kowiki dump origin is allowed')
    with urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=60) as response:
        return response.read()


def completed_articles_entry(payload, date):
    # Wikimedia may publish the aggregate in articlesdumprecombine rather than
    # the per-part articlesdump job. Match the exact aggregate filename.
    name = f'kowiki-{date}-pages-articles.xml.bz2'
    matches = [(job_name, job['files'][name]) for job_name, job in payload.get('jobs', {}).items()
               if job.get('status') == 'done' and name in job.get('files', {})]
    if len(matches) > 1:
        raise ValueError('Ambiguous completed aggregate dump metadata')
    return matches[0] if matches else None


def discover(root=ROOT):
    path = root / 'dump/manifest.json'
    if path.exists():
        return read(path)
    index = fetch('https://dumps.wikimedia.org/kowiki/').decode()
    dates = sorted(set(re.findall(r'href="(\d{8})/', index)), reverse=True)
    (root / 'dump').mkdir(parents=True, exist_ok=True)
    (root / 'dump/directory-index.html').write_text(index)
    for date in dates:
        payload = fetch(f'https://dumps.wikimedia.org/kowiki/{date}/dumpstatus.json')
        (root / 'dump' / f'dumpstatus-{date}.json').write_bytes(payload)
        obj = json.loads(payload)
        name = f'kowiki-{date}-pages-articles.xml.bz2'
        match = completed_articles_entry(obj, date)
        if match is None:
            continue
        job_name, entry = match
        checksum = entry.get('sha1')
        if not checksum:
            sums = fetch(f'https://dumps.wikimedia.org/kowiki/{date}/kowiki-{date}-sha1sums.txt').decode()
            (root / 'dump/sha1sums.txt').write_text(sums)
            checksum = next(line.split()[0] for line in sums.splitlines() if line.split()[-1] == name)
        (root / 'dump/dumpstatus.json').write_bytes(payload)
        value = {'dump_date': date, 'name': name, 'size_bytes': entry['size'], 'sha1': checksum,
                 'url': entry['url'] if entry['url'].startswith('https://') else 'https://dumps.wikimedia.org' + entry['url'],
                 'completed_job_name': job_name,
                 'discovered_utc': datetime.now(timezone.utc).isoformat(),
                 'latest_completed_articlesdump': True, 'official_origin': 'https://dumps.wikimedia.org/kowiki/',
                 'license': LICENSE, 'license_url': LICENSE_URL,
                 'licensing_source': 'https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content'}
        write(path, value)
        return value
    raise RuntimeError('No complete aggregate pages-articles dump in the official listing')


def download(root=ROOT):
    manifest = discover(root)
    expected_url = f"https://dumps.wikimedia.org/kowiki/{manifest['dump_date']}/{manifest['name']}"
    if manifest['url'] != expected_url:
        raise ValueError('The dump URL must be the frozen official dated artifact')
    destination = root / 'dump' / manifest['name']
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + '.part')
        status(root, 'download', dump=manifest['name'], max_download_bytes_per_second=8 * 1024 * 1024)
        subprocess.run(['curl', '-fL', '--retry', '3', '--retry-delay', '5', '--continue-at', '-',
                        '--limit-rate', '8M', '--user-agent', USER_AGENT, '--output', str(partial), manifest['url']], check=True)
        if partial.stat().st_size != manifest['size_bytes'] or digest(partial, 'sha1') != manifest['sha1']:
            raise ValueError('Downloaded dump size/SHA1 differs from Wikimedia manifest')
        partial.replace(destination)
    if destination.stat().st_size != manifest['size_bytes'] or digest(destination, 'sha1') != manifest['sha1']:
        raise ValueError('Saved dump checksum mismatch')
    value = {**manifest, 'local_path': str(destination), 'sha256': digest(destination), 'checksum_verified': True}
    write(root / 'dump/verified.json', value)
    status(root, 'download_complete', dump=manifest['name'], bytes=manifest['size_bytes'])
    return value


def extract(root=ROOT):
    from .wiki_local import iter_pages, page_filter, passages
    verified = read(root / 'dump/verified.json')
    contract = {'version': EXTRACTOR_VERSION, 'dump_sha256': verified['sha256'], 'minimum_paragraph_chars': 1,
                'extractor_code_sha256': digest(Path(__file__).with_name('wiki_local.py')),
                'driver_code_sha256': digest(Path(__file__)),
                'license': LICENSE, 'remove': ['redirect', 'disambiguation', 'list_page', 'non_article_namespace'],
                'markup_policy': 'no online template expansion; remove templates/tables/lists/refs; retain prose paragraphs'}
    frozen(root / 'extract_contract.json', contract)
    if (root / 'extraction.json').exists():
        return read(root / 'extraction.json')
    with sqlite3.connect(root / 'passages.sqlite') as db:
        db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS passages(doc_id INTEGER PRIMARY KEY, passage_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS progress(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        prior = db.execute("SELECT value FROM progress WHERE key='extract'").fetchone()
        progress = json.loads(prior[0]) if prior else {'pages_processed': 0, 'passages': 0, 'retained_articles': 0, 'drops': {}}
        counts = Counter(progress['drops']); start = progress['pages_processed']; doc_id = progress['passages']
        for index, page in enumerate(iter_pages(verified['local_path']), 1):
            if index <= start:
                continue
            reason = page_filter(page)
            if reason:
                counts[reason] += 1
            else:
                values = list(passages(page, verified['dump_date']))
                if not values:
                    counts['no_eligible_prose_paragraph'] += 1
                else:
                    progress['retained_articles'] += 1
                    for value in values:
                        db.execute('INSERT INTO passages VALUES (?,?,?)', (doc_id, value['passage_id'], json.dumps(value, ensure_ascii=False)))
                        doc_id += 1
            progress.update(pages_processed=index, passages=doc_id, drops=dict(counts))
            if index % 5000 == 0:
                db.execute("INSERT OR REPLACE INTO progress VALUES ('extract',?)", (json.dumps(progress),)); db.commit()
                status(root, 'extract', **progress)
        db.execute("INSERT OR REPLACE INTO progress VALUES ('extract',?)", (json.dumps(progress),)); db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    value = {**progress, 'contract': contract, 'complete': True}
    write(root / 'extraction.json', value)
    status(root, 'extraction_complete', **progress)
    return value


def tokenize(root=ROOT):
    from .wiki_local import KiwiTokenizer
    contract = {'version': TOKENIZER_VERSION, 'fields': ['title', 'section', 'text'], 'kiwi_workers': 2,
                'tokenizer_code_sha256': digest(Path(__file__).with_name('wiki_local.py')),
                'kiwipiepy': importlib.metadata.version('kiwipiepy'), 'bm25s': importlib.metadata.version('bm25s'),
                'mwparserfromhell': importlib.metadata.version('mwparserfromhell')}
    frozen(root / 'tokenizer_contract.json', contract)
    if (root / 'tokenization.json').exists():
        return read(root / 'tokenization.json')
    kiwi = KiwiTokenizer(workers=2)
    with sqlite3.connect(root / 'passages.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS tokens(doc_id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        done = db.execute('SELECT COALESCE(MAX(doc_id),-1)+1 FROM tokens').fetchone()[0]
        total = db.execute('SELECT COUNT(*) FROM passages').fetchone()[0]
        while done < total:
            rows = db.execute('SELECT doc_id,payload FROM passages WHERE doc_id>=? ORDER BY doc_id LIMIT 256', (done,)).fetchall()
            texts = [' '.join(json.loads(raw)[f] for f in contract['fields']) for _, raw in rows]
            for (doc_id, _), terms in zip(rows, kiwi.batch(texts)):
                db.execute('INSERT INTO tokens VALUES (?,?)', (doc_id, json.dumps(terms, ensure_ascii=False)))
                done = doc_id + 1
            db.commit()
            if done % 4096 == 0 or done == total:
                status(root, 'tokenize', completed=done, total=total)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    value = {'complete': True, 'passages': total, 'contract': contract}
    write(root / 'tokenization.json', value)
    return value


def index(root=ROOT):
    import bm25s
    from bm25s.tokenization import Tokenized
    contract = {'method': 'lucene', 'k1': 1.5, 'b': 0.75, 'dtype': 'float32', 'threads': 1,
                'tokenizer': read(root / 'tokenizer_contract.json'), 'extractor': read(root / 'extract_contract.json')}
    frozen(root / 'index_contract.json', contract)
    if (root / 'index_complete.json').exists():
        return read(root / 'index_complete.json')
    vocab = {}; documents = []
    with sqlite3.connect(f'file:{root / "passages.sqlite"}?mode=ro', uri=True) as db:
        for doc_id, raw in db.execute('SELECT doc_id,payload FROM tokens ORDER BY doc_id'):
            if doc_id != len(documents):
                raise ValueError('Non-contiguous document IDs')
            terms = json.loads(raw)
            documents.append([vocab.setdefault(term, len(vocab)) for term in terms])
        if len(documents) != db.execute('SELECT COUNT(*) FROM passages').fetchone()[0]:
            raise ValueError('Un-tokenized passage in corpus')
    status(root, 'index', passages=len(documents), vocabulary=len(vocab))
    engine = bm25s.BM25(method='lucene', k1=1.5, b=0.75)
    engine.index(Tokenized(ids=documents, vocab=vocab), show_progress=False)
    engine.save(str(root / 'bm25'))
    value = {'complete': True, 'passages': len(documents), 'vocabulary': len(vocab), 'contract': contract,
             'artifacts': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted((root / 'bm25').iterdir()) if p.is_file()},
             'database_bytes': (root / 'passages.sqlite').stat().st_size}
    write(root / 'index_complete.json', value)
    status(root, 'index_complete', passages=len(documents), vocabulary=len(vocab))
    return value


def main():
    constrain()
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['discover', 'download', 'extract', 'tokenize', 'index', 'build', 'search'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--query')
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    if args.command == 'build':
        for function in (download, extract, tokenize, index):
            function(args.root)
    elif args.command == 'search':
        from .wiki_local import LocalWikiSearch
        engine = LocalWikiSearch(args.root)
        try:
            print(json.dumps(engine.search(args.query), ensure_ascii=False, indent=2))
        finally:
            engine.close()
    else:
        print(json.dumps(globals()[args.command](args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
