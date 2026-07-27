"""
Обход сайтов производителей и поиск в интернете.

  python crawl.py sites                  — обойти сайты из crawl_sources.txt
  python crawl.py page <url>             — забрать одну страницу
  python crawl.py search "вопрос"        — поиск в интернете
  python crawl.py urls                   — вытащить ссылки из .url-файлов базы

Зачем. В базе 47 ярлыков на порталы производителей: по Wilo, IMP и Usystems
локальных документов нет вовсе, есть только ссылка. Плюс 67 страниц,
сохранённых вручную год назад и с тех пор устаревших.

Как это устроено правильно (и почему именно так):
  * плановый обход — основной путь. Раз в сутки или в неделю по списку
    источников, с записью в тот же индекс, но с пометкой источника;
  * поиск на лету — резерв для случая «в базе нет». Он медленный
    (рендеринг страницы плюс обращение к модели), поэтому удачные находки
    дописываются в плановый индекс, чтобы второй раз не искать;
  * каждый документ из интернета несёт source_type и дату обхода, и бот
    обязан показывать их в ответе. Смешивать «из базы» и «нашёл в сети»
    без пометки нельзя: доверие к этим источникам разное.

Перед тем как строить обход, стоит спросить у поставщиков официальные
выгрузки: YML-фид, обмен CommerceML или просто прайс по расписанию на
почту. Это надёжнее краулера и не создаёт правовых рисков по статье 1334
ГК РФ об извлечении существенной части базы данных.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

import config
import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_tables() -> None:
    db.connect().executescript("""
    CREATE TABLE IF NOT EXISTS web_pages (
        id           INTEGER PRIMARY KEY,
        url          TEXT UNIQUE,
        domain       TEXT,
        title        TEXT,
        text         TEXT,
        content_hash TEXT,
        etag         TEXT,
        last_modified TEXT,
        status       INTEGER,
        crawled_at   TEXT,
        changed_at   TEXT,
        doc_id       INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_web_domain ON web_pages(domain);
    """)


# ------------------------------------------------------------- загрузка -----
def _client():
    import httpx
    return httpx.Client(timeout=45, follow_redirects=True,
                        proxy=config.CRAWLER_PROXY or None,
                        headers={"User-Agent": config.CRAWL_USER_AGENT})


def fetch(url: str, etag: str | None = None, last_modified: str | None = None):
    """Возвращает (html, статус, etag, last-modified). 304 = страница не менялась."""
    headers = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    if config.CRAWL_RENDER_JS:
        return _fetch_rendered(url)
    with _client() as client:
        r = client.get(url, headers=headers)
        if r.status_code == 304:
            return None, 304, etag, last_modified
        return r.text, r.status_code, r.headers.get("etag"), r.headers.get("last-modified")


def _fetch_rendered(url: str):
    """Страница, собираемая скриптами. Требует playwright с браузером."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, 0, None, None
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True,
                                     proxy={"server": config.CRAWLER_PROXY}
                                     if config.CRAWLER_PROXY else None)
        page = browser.new_page(user_agent=config.CRAWL_USER_AGENT)
        page.goto(url, wait_until="networkidle", timeout=60000)
        # Догрузка каталогов с бесконечной прокруткой.
        for _ in range(6):
            before = page.evaluate("document.body.scrollHeight")
            page.mouse.wheel(0, 20000)
            page.wait_for_timeout(700)
            if page.evaluate("document.body.scrollHeight") == before:
                break
        html = page.content()
        browser.close()
    return html, 200, None, None


def to_text(html: str) -> tuple[str, str]:
    """Возвращает (заголовок, содержательный текст)."""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.S | re.I)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    try:
        import trafilatura
        text = trafilatura.extract(html, favor_recall=True, include_tables=True)
        if text:
            return title, text
    except ImportError:
        pass
    body = re.sub(r"(?is)<(script|style|nav|footer|header|noscript).*?</\1>", " ", html)
    body = re.sub(r"<[^>]+>", "\n", body)
    body = re.sub(r"&nbsp;?", " ", body)
    lines = [ln.strip() for ln in body.splitlines()]
    return title, "\n".join(ln for ln in lines if len(ln) > 2)


def links(html: str, base: str) -> tuple[set[str], set[str]]:
    """Возвращает (страницы того же домена, ссылки на PDF)."""
    pages, pdfs = set(), set()
    host = urllib.parse.urlparse(base).netloc
    for href in re.findall(r'href=["\']([^"\'#]+)', html, flags=re.I):
        url = urllib.parse.urljoin(base, href)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.path.lower().endswith(".pdf"):
            pdfs.add(url)
        elif parsed.netloc == host:
            pages.add(url.split("?")[0])
    return pages, pdfs


def sitemap_urls(root: str) -> list[str]:
    """Карта сайта — самый дешёвый способ узнать, что где лежит."""
    out: list[str] = []
    for candidate in ("/sitemap.xml", "/sitemap_index.xml"):
        try:
            with _client() as client:
                r = client.get(urllib.parse.urljoin(root, candidate))
            if r.status_code != 200:
                continue
            out += re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
        except Exception:  # noqa: BLE001
            continue
    return out


def allowed(url: str, cache: dict) -> bool:
    if not config.CRAWL_RESPECT_ROBOTS:
        return True
    host = urllib.parse.urlparse(url).netloc
    rp = cache.get(host)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"https://{host}/robots.txt")
        try:
            rp.read()
        except Exception:  # noqa: BLE001
            rp = None
        cache[host] = rp
    return True if rp is None else rp.can_fetch(config.CRAWL_USER_AGENT, url)


# ------------------------------------------------------------- индексация ---
def save_page(url: str, title: str, text: str) -> str:
    """Кладёт страницу в индекс как документ с пометкой источника."""
    import hashlib

    import chunk as chunker
    import embeddings
    import index as index_mod

    ensure_tables()
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    old = db.q1("SELECT id, content_hash, doc_id FROM web_pages WHERE url=?", (url,))
    if old and old["content_hash"] == content_hash:
        db.run("UPDATE web_pages SET crawled_at=? WHERE url=?", (_now(), url))
        return "unchanged"

    domain = urllib.parse.urlparse(url).netloc
    rel = f"web/{domain}{urllib.parse.urlparse(url).path or '/'}"[:400]
    existing = db.q1("SELECT id FROM documents WHERE rel_path=?", (rel,))
    if existing:
        index_mod._drop_document(existing["id"])
        doc_id = existing["id"]
        db.run("""UPDATE documents SET file_name=?, content_hash=?, text_chars=?,
                  indexed_at=?, status='ok', source_type='manufacturer_site',
                  source_url=?, effective_date=? WHERE id=?""",
               (title or domain, content_hash, len(text), _now(), url,
                _now()[:10], doc_id))
    else:
        cur = db.run("""INSERT INTO documents(rel_path, abs_path, file_name, ext, section,
                        brand, doc_type, content_hash, size_bytes, mtime, effective_date,
                        version_key, pages, text_chars, needs_ocr, indexed_at, status,
                        kind, source_type, source_url)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (rel, url, title or domain, ".html", "САЙТЫ", domain, "САЙТ",
                      content_hash, len(text), time.time(), _now()[:10],
                      f"web|{url}", 1, len(text), 0, _now(), "ok",
                      "web", "manufacturer_site", url))
        doc_id = int(cur.lastrowid)

    doc_meta = {"brand": domain, "doc_type": "страница сайта", "section": "САЙТЫ",
                "file_name": title or url, "effective_date": _now()[:10]}
    chunks = chunker.chunk_document([text], doc_meta)
    conn = db.connect()
    ids, texts = [], []
    for c in chunks:
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, ord, heading, context, text, n_chars) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, c.ord, c.heading, c.context, c.text, len(c.text)))
        ids.append(int(cur.lastrowid))
        texts.append(c.indexed_text)
    conn.commit()
    if ids:
        index_mod._fts_insert(ids)
        db.vectors().add(ids, embeddings.embed_texts(texts))
        db.vectors().save()

    db.run("""INSERT INTO web_pages(url, domain, title, text, content_hash, status,
              crawled_at, changed_at, doc_id)
              VALUES (?,?,?,?,?,?,?,?,?)
              ON CONFLICT(url) DO UPDATE SET title=excluded.title, text=excluded.text,
              content_hash=excluded.content_hash, crawled_at=excluded.crawled_at,
              changed_at=excluded.changed_at, doc_id=excluded.doc_id""",
           (url, domain, title, text, content_hash, 200, _now(), _now(), doc_id))
    return "indexed" if not old else "updated"


def crawl_site(root: str, max_pages: int | None = None) -> dict:
    max_pages = max_pages or config.CRAWL_MAX_PAGES
    ensure_tables()
    robots: dict = {}
    seen: set[str] = set()
    queue = [root] + sitemap_urls(root)[:max_pages]
    counts = {"indexed": 0, "updated": 0, "unchanged": 0, "skipped": 0, "error": 0}
    pdfs: set[str] = set()

    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if not allowed(url, robots):
            counts["skipped"] += 1
            continue
        row = db.q1("SELECT etag, last_modified FROM web_pages WHERE url=?", (url,))
        try:
            html, status, etag, lm = fetch(url, row["etag"] if row else None,
                                           row["last_modified"] if row else None)
        except Exception as exc:  # noqa: BLE001
            counts["error"] += 1
            print(f"  ошибка {url}: {exc}")
            time.sleep(config.CRAWL_DELAY_SECONDS)
            continue
        if status == 304 or not html:
            counts["unchanged"] += 1
        else:
            title, text = to_text(html)
            if len(text) > 200:
                counts[save_page(url, title, text)] += 1
            else:
                counts["skipped"] += 1
            page_links, page_pdfs = links(html, url)
            pdfs |= page_pdfs
            for link in page_links:
                if link not in seen and len(queue) < max_pages * 2:
                    queue.append(link)
            db.run("UPDATE web_pages SET etag=?, last_modified=? WHERE url=?", (etag, lm, url))
        time.sleep(config.CRAWL_DELAY_SECONDS)

    print(f"  {root}: {counts}, найдено PDF: {len(pdfs)}")
    return {"counts": counts, "pdfs": sorted(pdfs)}


def read_sources() -> list[str]:
    """Список адресов для обхода. Файл создаётся пустым при первом обращении."""
    src = config.CRAWL_SOURCES_FILE
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("# По одному адресу в строке. Строки с # игнорируются.\n"
                       "# https://wilo.ru/\n# https://imp-pump.ru/documents/\n",
                       encoding="utf-8")
        return []
    return [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def write_sources(lines: list[str]) -> None:
    config.CRAWL_SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CRAWL_SOURCES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sources(limit: int | None = None, progress=None) -> dict:
    """
    Обходит все источники из списка. Возвращает сводку по каждому.

    Перед запуском стоит спросить у поставщиков официальные выгрузки:
    это надёжнее обхода сайта и снимает правовые вопросы. Обход —
    запасной вариант для брендов, у которых локальной документации
    в базе почти нет, а на портале она есть.
    """
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    ensure_tables()
    roots = read_sources()
    if not roots:
        say(f"Список источников пуст: {config.CRAWL_SOURCES_FILE}")
        return {"sources": 0, "pages": 0, "results": []}
    say(f"Источников: {len(roots)}")
    results = []
    for i, root in enumerate(roots, 1):
        say(f"[{i}/{len(roots)}] обхожу {root}")
        try:
            stats = crawl_site(root, limit)
        except Exception as exc:  # noqa: BLE001 — один сайт не должен ронять обход
            say(f"    не вышло: {exc}")
            results.append({"root": root, "error": str(exc)})
            continue
        stats["root"] = root
        results.append(stats)
        say(f"    страниц: {stats.get('saved', 0)}")
    return {"sources": len(roots), "results": results,
            "pages": sum(r.get("saved", 0) for r in results)}


def cmd_sites(limit_pages: int | None) -> None:
    db.init()
    src = config.CRAWL_SOURCES_FILE
    if not src.exists():
        src.write_text("# По одному адресу в строке. Строки с # игнорируются.\n"
                       "# https://wilo.ru/\n# https://imp-pump.ru/documents/\n",
                       encoding="utf-8")
        print(f"Создан пустой список источников: {src}. Заполните его.")
        return
    roots = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    print(f"Источников: {len(roots)}")
    for root in roots:
        print(f"Обхожу {root}")
        crawl_site(root, limit_pages)


def cmd_urls() -> None:
    """Собирает адреса из ярлыков .url, лежащих в базе."""
    db.init()
    found = []
    for path in config.KB_ROOT.rglob("*.url"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(r"^URL=(.+)$", text, flags=re.M)
        if m:
            found.append((str(path.relative_to(config.KB_ROOT)), m.group(1).strip()))
    print(f"Ярлыков найдено: {len(found)}")
    domains: dict[str, int] = {}
    for rel, url in found:
        host = urllib.parse.urlparse(url).netloc
        domains[host] = domains.get(host, 0) + 1
    for host, n in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {n:>3}  {host}")
    out = config.CRAWL_SOURCES_FILE.with_name("crawl_sources_suggested.txt")
    out.write_text("\n".join(f"https://{h}/" for h in domains), encoding="utf-8")
    print(f"Предложенный список источников: {out}")


# ---------------------------------------------------------- поиск в сети ----
def web_search(query: str, limit: int = 5) -> list[dict]:
    """Поиск с ограничением по доменам партнёров, если он задан."""
    provider = config.WEB_SEARCH_PROVIDER
    if provider == "none":
        return []
    import httpx
    client = httpx.Client(timeout=30, proxy=config.CRAWLER_PROXY or None)
    domains = config.WEB_SEARCH_DOMAINS

    if provider == "yandex":
        # Yandex Search API: единственный вариант с рублёвой оплатой
        # и закрывающими документами для российского юрлица.
        text = query
        if domains:
            text += " " + " OR ".join(f"site:{d}" for d in domains)
        r = client.post(
            "https://searchapi.api.cloud.yandex.net/v2/web/search",
            headers={"Authorization": f"Api-Key {config.YANDEX_SEARCH_API_KEY}"},
            json={"query": {"searchType": "SEARCH_TYPE_RU", "queryText": text},
                  "folderId": config.YANDEX_SEARCH_FOLDER,
                  "responseFormat": "FORMAT_XML"})
        r.raise_for_status()
        import base64
        raw = base64.b64decode(r.json().get("rawData", "")).decode("utf-8", "ignore")
        out = []
        for m in re.finditer(r"<url>(.*?)</url>.*?<title>(.*?)</title>", raw, flags=re.S):
            out.append({"url": m.group(1), "title": re.sub(r"<[^>]+>", "", m.group(2))})
            if len(out) >= limit:
                break
        return out

    if provider == "searxng":
        r = client.get(f"{config.SEARXNG_URL}/search",
                       params={"q": query, "format": "json", "language": "ru"})
        r.raise_for_status()
        return [{"url": x["url"], "title": x.get("title", "")}
                for x in r.json().get("results", [])[:limit]]

    if provider == "tavily":
        payload = {"api_key": config.TAVILY_API_KEY, "query": query,
                   "max_results": limit, "search_depth": "basic"}
        if domains:
            payload["include_domains"] = domains
        r = client.post("https://api.tavily.com/search", json=payload)
        r.raise_for_status()
        return [{"url": x["url"], "title": x.get("title", "")}
                for x in r.json().get("results", [])]

    raise RuntimeError(f"неизвестный провайдер поиска: {provider}")


def cmd_search(query: str, index_results: bool) -> None:
    db.init()
    results = web_search(query)
    if not results:
        print("Поиск в интернете выключен (WEB_SEARCH_PROVIDER=none) или ничего не найдено.")
        return
    for r in results:
        print(f"  {r['title'][:70]}\n      {r['url']}")
        if index_results:
            try:
                html, status, _e, _l = fetch(r["url"])
                if html:
                    title, text = to_text(html)
                    print("      ->", save_page(r["url"], title, text))
            except Exception as exc:  # noqa: BLE001
                print(f"      не загрузилось: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description="Обход сайтов и поиск в интернете")
    p.add_argument("command", choices=["sites", "page", "search", "urls"])
    p.add_argument("args", nargs="*")
    p.add_argument("--max-pages", type=int)
    p.add_argument("--index", action="store_true", help="сохранять найденное в базу")
    a = p.parse_args()
    if a.command == "sites":
        cmd_sites(a.max_pages)
    elif a.command == "urls":
        cmd_urls()
    elif a.command == "page":
        db.init()
        html, status, _e, _l = fetch(a.args[0])
        title, text = to_text(html or "")
        print(f"{status} {title}\n{len(text)} символов")
        if a.index:
            print(save_page(a.args[0], title, text))
        else:
            print(text[:1500])
    elif a.command == "search":
        cmd_search(" ".join(a.args), a.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
