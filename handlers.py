"""
Обработчики задач очереди.

Здесь собрано в одном месте всё, что можно запустить кнопкой из админки
или командой из очереди. Каждый обработчик получает параметры и функцию
для сообщений о ходе работы, а возвращает то, что попадёт в результат
задачи и будет показано в интерфейсе.

Отдельный модуль нужен, чтобы очередь ничего не знала о предметной
области, а предметные модули — об очереди.
"""
from __future__ import annotations

from pathlib import Path

import config
import jobs


@jobs.handler("reindex")
def _reindex(payload: dict, progress) -> dict:
    import index as index_mod
    progress("Обхожу папку базы знаний")
    return index_mod.build(force=False, limit=payload.get("limit"))


@jobs.handler("reindex_full")
def _reindex_full(payload: dict, progress) -> dict:
    import index as index_mod
    progress("Полная переиндексация: разбираются все файлы заново")
    result = index_mod.build(force=True, limit=payload.get("limit"))
    _regression_after("полная переиндексация", progress, result)
    return result


def _regression_after(reason: str, progress, result: dict) -> None:
    """
    Прогоняет контрольные вопросы сразу после изменения индекса.

    Иначе падение качества обнаруживается через недели по жалобам, и
    связать его с конкретной переиндексацией уже нельзя. Если набора
    вопросов нет, молча пропускаем: это не ошибка задачи.
    """
    try:
        import regression
        if not regression.dataset_path().exists():
            progress("Набор контрольных вопросов не найден — проверка качества "
                     "пропущена. Без него любое изменение делается вслепую.")
            return
        progress("Прогоняю контрольные вопросы, чтобы увидеть, "
                 "не ухудшилось ли качество")
        run = regression.run(reason=reason, progress=progress)
        result["regression"] = {"hit": run["hit"], "mrr": run["mrr"],
                                "delta": run["delta"]}
        if run.get("delta") and run["delta"]["mrr"] <= -0.03:
            progress(f"ВНИМАНИЕ: качество поиска упало "
                     f"(MRR {run['delta']['mrr']:+}). Раздел «Аналитика» → "
                     f"«Проверки качества» покажет, какие вопросы сломались.")
    except Exception as exc:  # noqa: BLE001 — проверка не должна ронять задачу
        progress(f"Проверку качества выполнить не удалось: {exc}")


@jobs.handler("repair")
def _repair(payload: dict, progress) -> dict:
    import index as index_mod
    progress("Досчитываю недостающие векторы")
    return {"chunks": index_mod.repair()}


@jobs.handler("train_lsa")
def _train_lsa(payload: dict, progress) -> dict:
    import embeddings
    progress("Обучаю смысловую модель на текущем содержимом индекса")
    model = embeddings.LSAEmbedder.train(dim=payload.get("dim"), progress=progress)
    embeddings.reset()
    return {"documents": model.meta.get("documents"), "vocab": model.meta.get("vocab"),
            "dim": model.dim, "seconds": model.meta.get("trained_seconds")}


@jobs.handler("reembed")
def _reembed(payload: dict, progress) -> dict:
    import index as index_mod
    provider = payload.get("provider") or config.EMBEDDINGS_PROVIDER
    progress(f"Пересчитываю векторы провайдером {provider}")
    n = index_mod.reembed(provider=payload.get("provider"),
                          only_missing=bool(payload.get("only_missing")),
                          progress=progress)
    result = {"chunks": n, "provider": provider}
    _regression_after(f"пересчёт векторов ({provider})", progress, result)
    return result


@jobs.handler("ocr")
def _ocr(payload: dict, progress) -> dict:
    import ocr as ocr_mod
    progress("Распознаю сканы")
    return ocr_mod.run(limit=payload.get("limit"), provider=payload.get("provider"),
                       progress=progress)


@jobs.handler("ocr_retry")
def _ocr_retry(payload: dict, progress) -> dict:
    import ocr as ocr_mod
    progress("Повторяю документы, на которых была ошибка")
    return ocr_mod.run(limit=payload.get("limit"), retry_failed=True, progress=progress)


@jobs.handler("backup")
def _backup(payload: dict, progress) -> dict:
    import backup as backup_mod
    archive = backup_mod.create(note=payload.get("note", ""), progress=progress)
    return {"archive": archive.name, "bytes": archive.stat().st_size}


@jobs.handler("backup_verify")
def _backup_verify(payload: dict, progress) -> dict:
    import backup as backup_mod
    name = payload.get("name")
    target = next((a for a in backup_mod.archives()
                   if not name or a.name == Path(str(name)).name), None)
    if target is None:
        raise jobs.JobError("копий пока нет")
    progress(f"Разворачиваю {target.name} во временную папку")
    return backup_mod.verify_archive(target)


@jobs.handler("backup_prune")
def _backup_prune(payload: dict, progress) -> dict:
    import backup as backup_mod
    removed = backup_mod.prune(quiet=True)
    progress(f"Удалено копий: {len(removed)}")
    return {"removed": removed}


@jobs.handler("restore")
def _restore(payload: dict, progress) -> dict:
    import backup as backup_mod
    name = Path(str(payload.get("name", ""))).name
    target = next((a for a in backup_mod.archives() if a.name == name), None)
    if target is None:
        raise jobs.JobError(f"копия «{name}» не найдена")
    progress(f"Восстанавливаю из {target.name}")
    return backup_mod.restore(target, force=bool(payload.get("force")))


@jobs.handler("graph")
def _graph(payload: dict, progress) -> dict:
    import graph
    progress("Строю граф связности")
    g = graph.build_graph()
    graph.render_html(g, config.DATA_DIR / "graph.html")
    return g["stats"]


@jobs.handler("structure")
def _structure(payload: dict, progress) -> dict:
    import watcher
    progress("Сверяю структуру каталогов")
    return {"events": watcher.diff_structure()}


@jobs.handler("compare")
def _compare(payload: dict, progress) -> dict:
    import evaluate as eval_mod
    dataset = Path(payload.get("dataset") or "eval/golden.jsonl")
    if not dataset.exists():
        raise jobs.JobError(f"нет файла {dataset} — соберите набор контрольных вопросов")
    return eval_mod.compare(eval_mod.load(dataset), config.SEARCH_TOP_K, progress=progress)


@jobs.handler("eval_llm")
def _eval_llm(payload: dict, progress) -> dict:
    """
    Полный замер: не только «нашёлся ли документ», но и сам ответ —
    есть ли в нём ожидаемые цифры и нет ли запрещённых (подмена соседней
    моделью). Долгий: по обращению к модели на каждый вопрос, поэтому
    очередь пропускает вперёд живые вопросы сотрудников.
    """
    import evaluate as eval_mod
    dataset = Path(payload.get("dataset") or "eval/golden.jsonl")
    if not dataset.exists():
        raise jobs.JobError(f"нет файла {dataset} — соберите набор контрольных вопросов")
    data = eval_mod.load(dataset)
    progress(f"Прогоняю {len(data)} вопросов с генерацией — это долго")
    result = eval_mod.evaluate(data, config.SEARCH_TOP_K, run_llm=True)
    details = result.pop("details", [])
    result["misses"] = [d["question"] for d in details if d["rank"] is None][:20]
    result["substituted_questions"] = [
        {"question": d["question"], "by": d.get("substituted_by")}
        for d in details if d.get("substituted_by")][:20]
    result["forbidden"] = [
        {"question": d["question"], "found": d.get("forbidden_in_answer")}
        for d in details if d.get("forbidden_in_answer")][:20]
    return result


@jobs.handler("regression")
def _regression(payload: dict, progress) -> dict:
    import regression
    return regression.run(reason=payload.get("reason", "вручную"), progress=progress)


@jobs.handler("alerts")
def _alerts(payload: dict, progress) -> dict:
    import alerts
    progress("Проверяю, нет ли проблем, требующих внимания")
    return alerts.check()


@jobs.handler("retention")
def _retention(payload: dict, progress) -> dict:
    import retention
    progress("Удаляю данные, у которых вышел срок хранения")
    return retention.clean()


@jobs.handler("restore_drill")
def _restore_drill(payload: dict, progress) -> dict:
    """
    Учебное восстановление: копия разворачивается целиком и по ней
    прогоняются контрольные вопросы.

    Проверка при создании копии смотрит, что архив цел. Здесь проверяется
    другое — что развёрнутый индекс действительно работает. Это разные
    вещи, и вторая обнаруживает то, чего первая не видит.
    """
    import backup
    import tempfile
    from pathlib import Path as _P
    # Восстанавливаться в аварию придётся из ЗЕРКАЛА — локальные копии
    # погибнут вместе с диском. Значит, и учиться восстанавливаться надо
    # на зеркальной копии: локальная — лишь запасной вариант для
    # установок, где зеркало не настроено.
    target = None
    source = "локальная копия"
    import config as _cfg
    if _cfg.BACKUP_MIRROR_DIR:
        mirror = _P(_cfg.BACKUP_MIRROR_DIR).expanduser()
        mirrored = sorted(mirror.glob("*.tar.gz"),
                          key=lambda f: f.stat().st_mtime, reverse=True) \
            if mirror.exists() else []
        if mirrored:
            target, source = mirrored[0], "зеркало"
        else:
            progress("Зеркало настроено, но копий в нём нет — проверяю локальную")
    if target is None:
        archives = backup.archives()
        if not archives:
            raise jobs.JobError("копий нет")
        target = archives[0]
    progress(f"Разворачиваю {target.name} ({source}) во временную папку")
    report = backup.verify_archive(target)
    if not report["ok"]:
        raise jobs.JobError(f"копия не разворачивается: {report['error']}")

    import config as cfg
    import db
    import regression
    if not regression.dataset_path().exists():
        progress("Набор контрольных вопросов не найден — проверена только "
                 "целостность копии, но не работоспособность индекса из неё.")
        return {"archive": target.name, "verified": True, "searched": False,
                **report["counts"]}

    work = backup.workdir("kb_drill_")
    saved_data, saved_db = cfg.DATA_DIR, cfg.DB_PATH
    try:
        import tarfile
        with tarfile.open(target) as tar:
            backup._safe_extract(tar, work)
        cfg.DATA_DIR, cfg.DB_PATH = work, work / "kb.sqlite3"
        cfg.VECTORS_PATH, cfg.VECTOR_IDS_PATH = work / "vectors.npy", work / "vector_ids.json"
        cfg.LSA_MODEL_PATH = work / "lsa_model.npz"
        db._local.conn = None
        db.reset_vectors()
        import embeddings
        embeddings.reset()
        progress("Прогоняю контрольные вопросы по восстановленному индексу")
        result = regression.run(reason=f"учебное восстановление {target.name}",
                                progress=progress)
        return {"archive": target.name, "verified": True, "searched": True,
                "hit": result["hit"], "mrr": result["mrr"]}
    finally:
        cfg.DATA_DIR, cfg.DB_PATH = saved_data, saved_db
        cfg.VECTORS_PATH = saved_data / "vectors.npy"
        cfg.VECTOR_IDS_PATH = saved_data / "vector_ids.json"
        cfg.LSA_MODEL_PATH = saved_data / "lsa_model.npz"
        db._local.conn = None
        db.reset_vectors()
        import embeddings
        embeddings.reset()
        import shutil as _sh
        _sh.rmtree(work, ignore_errors=True)


@jobs.handler("crawl")
def _crawl(payload: dict, progress) -> dict:
    import crawl
    return crawl.run_sources(limit=payload.get("limit"), progress=progress)


@jobs.handler("contextual")
def _contextual(payload: dict, progress) -> dict:
    import contextual
    return contextual.run(limit=payload.get("limit"), progress=progress)


@jobs.handler("media")
def _media(payload: dict, progress) -> dict:
    import media
    return media.transcribe_queue(limit=payload.get("limit"), progress=progress)


@jobs.handler("model_install")
def _model_install(payload: dict, progress) -> dict:
    """
    Загрузка весов модели.

    Через очередь, а не фоновым потоком из веб-обработчика. Раньше поток
    был демоном и учитывался только в памяти процесса: перезапуск админки
    терял и загрузку, и всякую память о ней, а в папке оставались неполные
    веса, которые система считала готовой моделью. Дочерний процесс
    загрузчика при этом переживал остановку и продолжал писать в ту же
    папку уже после перезапуска.
    """
    import models as models_mod
    model_id = str(payload.get("id") or "")
    if not model_id:
        raise ValueError("не указана модель")
    return models_mod.install(model_id, engine=payload.get("engine") or None,
                              progress=progress)
