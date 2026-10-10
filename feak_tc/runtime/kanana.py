"""Keep the existing Kanana runtime's CUDA visibility changes process-local."""

import multiprocessing


def _worker(connection, kwargs):
    from feak_tc.diagnose import get_diagnoser

    try:
        diagnoser = get_diagnoser("kanana", **kwargs)
        while True:
            text = connection.recv()
            if text is None:
                return
            try:
                connection.send((True, diagnoser.diagnose(text)))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except EOFError:
        pass
    finally:
        connection.close()


class KananaWorker:
    """Lazy persistent worker: one trained scorer load, independent generator GPU."""

    def __init__(self, *, timeout_s, **kwargs):
        self.timeout_s = timeout_s
        self.kwargs = kwargs
        self._process = self._connection = None

    def diagnose(self, text):
        if self._process is None:
            context = multiprocessing.get_context("spawn")
            self._connection, child = context.Pipe()
            self._process = context.Process(target=_worker, args=(child, self.kwargs))
            self._process.start()
            child.close()
        try:
            self._connection.send(text)
            if not self._connection.poll(self.timeout_s):
                raise RuntimeError(f"Kanana diagnosis exceeded {self.timeout_s} seconds")
            ok, result = self._connection.recv()
            if not ok:
                raise RuntimeError(result)
            return result
        except (EOFError, BrokenPipeError) as exc:
            raise RuntimeError("Kanana worker exited before returning a diagnosis") from exc

    def close(self):
        if self._process is None:
            return
        try:
            self._connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()
        self._process = self._connection = None
