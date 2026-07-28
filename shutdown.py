"""
Корректная остановка: один сигнал — один порядок действий.

Зачем отдельный модуль. Систему останавливают постоянно и буднично:
`systemctl restart` при обновлении, `docker compose up -d` с новой
версией, перезагрузка сервера ночью. До сих пор ни один процесс не
обрабатывал сигнал остановки, и это означало ровно одно: остановка
всегда происходила в произвольной точке — в том числе посреди записи
векторного индекса или посреди упаковки резервной копии.

Особенно неприятно это в контейнере. Python там становится процессом
номер один, а для процесса номер один ядро не применяет действие по
умолчанию: SIGTERM просто игнорируется. `docker stop` честно ждёт
десять секунд и убивает процесс насмерть — каждый раз, при каждом
штатном обновлении. То есть «редкий случай аварии» на самом деле был
нормой.

Что здесь есть. Регистрация обработчика сигнала, флаг «пора
останавливаться», который длительные операции проверяют в своих циклах,
и список действий, выполняемых на выходе, — в порядке, обратном
регистрации.

Как этим пользоваться в длительной операции:

    import shutdown
    for i, file in enumerate(files):
        if shutdown.stopping():
            log.warning("остановка: обработано %d из %d", i, len(files))
            break
        ...

Смысл именно в том, чтобы прерваться в точке, где данные согласованы, а
не там, где застал сигнал.
"""
from __future__ import annotations

import os
import signal
import threading
import time

import logging_setup

log = logging_setup.get("web")

_stop = threading.Event()
_actions: list[tuple[str, object]] = []
_installed = False
_lock = threading.Lock()


def reset() -> None:
    """
    Забыть о запрошенной остановке. Нужно тестам: флаг глобальный на
    процесс, и тест, проверяющий остановку, иначе выключает длительные
    операции во всех последующих тестах — те завершаются мгновенно и
    «ничего не находят».
    """
    global _finisher
    _stop.clear()
    _finisher = None
    with _lock:
        _actions.clear()


def stopping() -> bool:
    """Попросили ли нас остановиться."""
    return _stop.is_set()


def wait(seconds: float) -> bool:
    """
    Пауза, прерываемая сигналом остановки.

    Обычный `time.sleep` в фоновом цикле означает, что остановка ждёт
    столько же: цикл с паузой в минуту задерживает выключение на минуту,
    и docker успевает перейти к SIGKILL.
    """
    return _stop.wait(seconds)


def on_stop(name: str, action) -> None:
    """Что сделать при остановке. Выполняется в обратном порядке."""
    with _lock:
        _actions.append((name, action))


def request_stop(reason: str = "") -> None:
    """Попросить остановку изнутри — например, при неустранимой ошибке."""
    if not _stop.is_set():
        log.warning("запрошена остановка%s", f": {reason}" if reason else "")
    _stop.set()


def run_actions(timeout: float = 20.0) -> None:
    """
    Выполняет отложенные действия. Каждое — в отдельном потоке с
    ограничением по времени.

    Отдельный поток здесь не для скорости, а чтобы не получить взаимную
    блокировку. Обработчик сигнала выполняется в главном потоке, а
    типичное действие на остановке — «перестать принимать запросы», то
    есть `shutdown()` у http-сервера. Этот вызов ждёт, пока завершится
    цикл приёма запросов, а цикл крутится как раз в главном потоке,
    который сейчас занят обработчиком сигнала. Получается ожидание
    самого себя: процесс висит до тех пор, пока его не убьют насмерть, —
    ровно то, чего мы избегали.

    Ограничение по времени нужно по той же причине: одно подвисшее
    действие не должно превращать корректную остановку в SIGKILL.
    """
    deadline = time.time() + timeout
    with _lock:
        actions = list(reversed(_actions))
        _actions.clear()
    for name, action in actions:
        left = deadline - time.time()
        if left <= 0:
            log.error("на остановке не хватило времени: «%s» не выполнено", name)
            continue
        started = time.time()
        error: list[BaseException] = []

        def run(fn=action, box=error):
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 — сообщим и пойдём дальше
                box.append(exc)

        worker = threading.Thread(target=run, name=f"stop:{name}", daemon=True)
        worker.start()
        worker.join(left)
        if worker.is_alive():
            log.error("остановка: «%s» не уложилось в %.0f с — продолжаю без него",
                      name, left)
        elif error:
            log.error("остановка: «%s» не выполнено (%s)", name, error[0])
        else:
            log.info("остановка: %s — готово за %.1f с", name, time.time() - started)


_finisher: threading.Thread | None = None


def install(name: str = "процесс", timeout: float = 20.0) -> None:
    """
    Ставит обработчики SIGTERM и SIGINT.

    Обработчик делает две вещи и сразу возвращается: поднимает флаг и
    запускает завершение в отдельном потоке. Быстрый возврат здесь
    обязателен. Обработчик сигнала выполняется в главном потоке, а
    главный поток обычно занят основным циклом — приёмом HTTP-запросов
    или опросом Telegram. Пока обработчик не вернётся, цикл не сдвинется
    с места, и любое действие, которое ждёт завершения этого цикла,
    будет ждать вечно. На практике это выглядело так: процесс получает
    сигнал, пишет в журнал «завершаю работу» и висит, пока его не убьют
    насмерть — то есть корректная остановка не работала именно там, где
    была нужна.

    Второй сигнал не откладывается: если человек нажал Ctrl+C дважды, он
    хочет выйти сейчас, а не досматривать, как мы аккуратно завершаемся.
    """
    global _installed, _finisher
    if _installed:
        return
    _installed = True
    hard = {"count": 0}

    def handler(signum, _frame):  # noqa: ANN001
        global _finisher
        hard["count"] += 1
        if hard["count"] > 1:
            log.error("повторный сигнал — выхожу немедленно")
            os._exit(130)
        log.warning("получен сигнал %s: %s завершает работу", signum, name)
        _stop.set()
        _finisher = threading.Thread(target=run_actions, args=(timeout,),
                                     name="shutdown", daemon=True)
        _finisher.start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Не главный поток — обработчик поставит тот, кто главный.
            pass


def wait_actions(timeout: float = 25.0) -> None:
    """Дождаться завершения отложенных действий — вызывается в конце main."""
    worker = _finisher
    if worker is not None:
        worker.join(timeout)
