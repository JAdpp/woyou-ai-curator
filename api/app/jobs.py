"""Async curation jobs with a visitor-visible step list.

The P0 build generated synchronously and drove a fake progress bar off a
percentage the backend never actually reported. This replaces that with real
steps: the generator emits a finding as each stage lands, the job store records
it, and the client reads the list over SSE (with polling as a fallback for
proxies that buffer event streams).

Jobs live in memory. They are cheap to lose — a failed job just means the
visitor generates again — so they deliberately do not go through the persisted
exhibition store.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import ssl
from time import perf_counter
from typing import Awaitable, Callable
from uuid import uuid4

from .collections import CollectionDataError
from .models import GenerationJob, JobStep, JobStepStatus, utc_now
from .providers.deepseek import ProviderError

# Ordered pipeline shown to the visitor. `poster` runs concurrently with the
# text stages but is listed in narrative order.
STEP_DEFINITIONS: list[tuple[str, str, str]] = [
    ("profile", "归纳你的需求", "把访谈整理成一份策展简报"),
    ("retrieve", "检索候选藏品", "在开放馆藏中筛选可用的展品"),
    ("theme", "确定展览主题", "为这次参观定一个主题与叙事线"),
    ("chapters", "划分章节、分配展品", "把展品编排成连续展线上的叙事区段"),
    ("labels", "撰写展签与引导语", "让每句说明都能回到机构记录"),
    ("poster", "生成展览海报", "为展厅入口生成一张横版视觉"),
    ("epilogue", "撰写结语与材料边界", "说清这场展览没有回答什么"),
    ("space", "编排 3D 展厅空间", "用连续动线连接叙事区段、灯光与画框"),
]

# The pipeline is on screen for the whole generation, so it is the most visible
# server-authored text there is.
STEP_DEFINITIONS_EN: list[tuple[str, str, str]] = [
    ("profile", "Read what you asked for", "Turn the interview into a curatorial brief"),
    ("retrieve", "Search for candidates", "Filter the open collections for usable objects"),
    ("theme", "Settle the theme", "Choose a subject and a line of argument for this visit"),
    ("chapters", "Chapter and assign", "Arrange the objects into segments along one route"),
    ("labels", "Write labels and lead-ins", "Keep every sentence traceable to an institutional record"),
    ("poster", "Make the poster", "Generate a landscape visual for the entrance"),
    ("epilogue", "Write the closing and its limits", "State what this exhibition does not answer"),
    ("space", "Lay out the 3D hall", "Link segments, lighting and frames along one path"),
]

MAX_JOB_SECONDS = 180.0


logger = logging.getLogger(__name__)


def _new_steps(language: str = "zh") -> list[JobStep]:
    definitions = STEP_DEFINITIONS_EN if language == "en" else STEP_DEFINITIONS
    return [
        JobStep(key=key, title=title, detail=detail)
        for key, title, detail in definitions
    ]


class JobStore:
    """In-memory job registry with per-job change notification."""

    def __init__(self, max_job_seconds: float = MAX_JOB_SECONDS) -> None:
        if max_job_seconds <= 0:
            raise ValueError("max_job_seconds must be positive")
        self.max_job_seconds = float(max_job_seconds)
        self._jobs: dict[str, GenerationJob] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def create(self, language: str = "zh") -> GenerationJob:
        job = GenerationJob(id=str(uuid4()), status="queued", steps=_new_steps(language))
        self._jobs[job.id] = job
        self._events[job.id] = asyncio.Event()
        return job

    def get(self, job_id: str) -> GenerationJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _notify(self, job_id: str) -> None:
        event = self._events.get(job_id)
        if event is not None:
            event.set()

    async def wait_for_change(self, job_id: str, timeout: float = 20.0) -> None:
        event = self._events.get(job_id)
        if event is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), timeout=timeout)
        event.clear()

    # -- mutation ---------------------------------------------------------

    def start_step(self, job_id: str, key: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        for step in job.steps:
            if step.key == key:
                step.status = JobStepStatus.RUNNING
                step.started_at = utc_now()
                job.stage = step.title
        job.status = "running"
        job.updated_at = utc_now()
        self._notify(job_id)

    def finish_step(self, job_id: str, key: str, finding: str | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        for step in job.steps:
            if step.key == key:
                step.status = JobStepStatus.DONE
                step.finding = finding
                step.finished_at = utc_now()
        self._recompute_progress(job)
        self._notify(job_id)

    def skip_step(self, job_id: str, key: str, finding: str | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        for step in job.steps:
            if step.key == key:
                step.status = JobStepStatus.SKIPPED
                step.finding = finding
                step.finished_at = utc_now()
        self._recompute_progress(job)
        self._notify(job_id)

    def complete(self, job_id: str, exhibition_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        for step in job.steps:
            if step.status in {JobStepStatus.PENDING, JobStepStatus.RUNNING}:
                step.status = JobStepStatus.DONE
                step.finished_at = utc_now()
        job.status = "completed"
        job.progress = 100
        job.stage = "完成"
        job.exhibition_id = exhibition_id
        job.updated_at = utc_now()
        self._notify(job_id)

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        error_code: str = "curation_failed",
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        for step in job.steps:
            if step.status == JobStepStatus.RUNNING:
                step.status = JobStepStatus.FAILED
                step.finished_at = utc_now()
        job.status = "failed"
        job.error_code = error_code
        job.error = error
        job.stage = "生成未完成"
        job.updated_at = utc_now()
        self._notify(job_id)

    @staticmethod
    def _recompute_progress(job: GenerationJob) -> None:
        settled = sum(
            1
            for step in job.steps
            if step.status in {JobStepStatus.DONE, JobStepStatus.SKIPPED}
        )
        job.progress = min(99, int(settled / max(1, len(job.steps)) * 100))
        job.updated_at = utc_now()

    # -- execution --------------------------------------------------------

    def run(
        self,
        job_id: str,
        work: Callable[["JobStore", str], Awaitable[None]],
    ) -> None:
        """Schedule ``work`` and guarantee the job reaches a terminal state."""

        async def runner() -> None:
            started = perf_counter()
            try:
                await asyncio.wait_for(
                    work(self, job_id), timeout=self.max_job_seconds
                )
            except asyncio.TimeoutError:
                job = self._jobs.get(job_id)
                stage = job.stage if job is not None else "unknown"
                logger.warning(
                    "curation job %s hit the %.1fs safety deadline at stage=%s",
                    job_id,
                    self.max_job_seconds,
                    stage,
                )
                self.fail(
                    job_id,
                    "策展服务响应过慢，本次任务已安全停止。请重试；这不代表主题不受支持。",
                    error_code="curation_timeout",
                )
            except asyncio.CancelledError:
                self.fail(
                    job_id,
                    "本次策展任务已停止。你的问题仍然保留，可以直接重试。",
                    error_code="curation_cancelled",
                )
                raise
            except ProviderError as error:
                self.fail(
                    job_id,
                    error.public_message,
                    error_code=error.code,
                )
            except CollectionDataError as error:
                # Collection errors carry precise internal details, but those
                # details are not a recovery instruction for a visitor.
                if error.code in {
                    "QUESTION_UNSUPPORTED",
                    "QUESTION_UNSUPPORTED_AFTER_AUDIT",
                }:
                    message = (
                        "当前馆藏不足以可靠回答这个问题。你的原问题仍然保留；"
                        "请选择系统建议的相近方向，或减少必须覆盖的地区后重试。"
                    )
                elif error.code in {
                    "RETRIEVAL_AUDIT_UNAVAILABLE",
                    "RETRIEVAL_AUDIT_INVALID",
                    "RETRIEVAL_SEARCH_TIMEOUT",
                }:
                    message = (
                        "证据检索或相关性审查暂时没有完成。你的问题仍然保留，"
                        "这不代表馆藏不支持该主题；请直接重试。"
                    )
                else:
                    message = (
                        "馆藏数据暂时无法完成本次策展。你的问题仍然保留，请稍后重试。"
                    )
                self.fail(
                    job_id,
                    message,
                    error_code=error.code.casefold(),
                )
            except (ssl.SSLError, OSError):
                # Defence in depth for any future provider that forgets to
                # normalise its transport errors at its own boundary.
                self.fail(
                    job_id,
                    "策展服务暂时无法连接。你的问题仍然保留，请稍后重试。",
                    error_code="provider_network_error",
                )
            except Exception:  # noqa: BLE001 - intentionally sanitised below
                logger.exception("curation job %s failed unexpectedly", job_id)
                self.fail(
                    job_id,
                    "生成未完成。你的问题仍然保留，请重试；若再次失败，可先缩短参观时长。",
                    error_code="curation_failed",
                )
            finally:
                job = self._jobs.get(job_id)
                logger.info(
                    "curation job %s settled status=%s elapsed=%.3fs",
                    job_id,
                    job.status if job is not None else "missing",
                    perf_counter() - started,
                )
                self._tasks.pop(job_id, None)

        self._tasks[job_id] = asyncio.create_task(runner())

    def emitter(self, job_id: str) -> Callable[[str, str], Awaitable[None]]:
        """Adapter matching :data:`generator.StepEmitter`.

        Marks the named step done and opens the next pending one, so the
        generator only has to report what it finished.
        """

        async def emit(key: str, finding: str) -> None:
            self.finish_step(job_id, key, finding)
            job = self._jobs.get(job_id)
            if job is None:
                return
            for step in job.steps:
                if step.status == JobStepStatus.PENDING and step.key != "poster":
                    self.start_step(job_id, step.key)
                    break
            # Yield so SSE subscribers observe each step separately rather than
            # receiving one batched update at the end.
            await asyncio.sleep(0)

        return emit
