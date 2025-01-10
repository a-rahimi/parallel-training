from typing import Callable, Dict, List, NamedTuple, Set, Tuple

import collections
from enum import Enum
import itertools

from IPython import display
import matplotlib
import numpy as np
import pandas as pd


Batch = int
Stage = int
Worker = int
Timestamp = int


class Direction(Enum):
    FORWARD = 1
    BACKWARD = -1


class Work(NamedTuple):
    stage: Stage
    batch: Batch
    direction: Direction

    def is_final(self) -> bool:
        return self.direction == Direction.BACKWARD and self.stage == 0

    def duration(self) -> int:
        return 1 if self.direction == Direction.FORWARD else 2

    def next(self, num_stages: int) -> "Work":
        if self.is_final():
            return self
        elif self.direction == Direction.FORWARD and self.stage == num_stages - 1:
            return Work(self.stage, self.batch, Direction.BACKWARD)
        elif self.direction == Direction.FORWARD:
            return Work(self.stage + 1, self.batch, Direction.FORWARD)
        elif self.direction == Direction.BACKWARD:
            return Work(self.stage - 1, self.batch, Direction.BACKWARD)

        assert False, "Unadvanceable work"

    def __str__(self) -> str:
        op = "f" if self.direction == Direction.FORWARD else "∇"
        return f"{op}{self.stage}"


class ComputeAndWeightWorkers(NamedTuple):
    compute_worker: Worker
    weight_worker: Worker


class PriorityOrder:
    def __init__(self, num_stages: int, num_batches: int):
        self.order: Dict[Work, int] = {}

    def __call__(self, work: Work) -> int:
        return self.order[work]


class BackwardFirst(PriorityOrder):
    def __init__(self, num_stages: int, num_batches: int):
        self.order = {
            Work(stage, batch, direction): n
            for n, (stage, batch, direction) in enumerate(
                itertools.product(range(-1, num_stages), range(num_batches), Direction)
            )
        }


class OldestBatchFirst(PriorityOrder):
    def __init__(self, num_stages: int, num_batches: int):
        self.order = {
            Work(stage, batch, direction): n
            for n, (batch, direction, stage) in enumerate(
                itertools.product(range(num_batches), Direction, range(-1, num_stages))
            )
        }


class WorkHistory(NamedTuple):
    work: Work
    start_time: Timestamp
    end_time: Timestamp
    performed_by: Worker


class SimulationStats:
    def __init__(
        self,
        num_workers: int,
        schedule: Callable[[Stage, Batch], ComputeAndWeightWorkers],
    ) -> None:
        self.num_workers = num_workers
        self.schedule = schedule

        self.all_work: List[WorkHistory] = []
        self.df_worker_stats = pd.DataFrame(
            0,
            index=range(num_workers),
            columns=[
                "Weight transmissions",
                "Weight storage",
                "Activation transmissions",
                "Activation storage",
                "Max activation storage",
            ],
            dtype=int,
        )
        self.df_worker_stats.index.name = "Worker"

        self.worker_weight_storage: Dict[Worker, Set[Stage]] = collections.defaultdict(
            set
        )

    def add(
        self,
        pred_work: Work,
        new_work: Work,
        start_time: Timestamp,
        end_time: Timestamp,
    ) -> None:
        pred_worker = self.schedule(pred_work).compute_worker
        new_worker, weight_worker = self.schedule(new_work)

        # Record the work.
        self.all_work.append(WorkHistory(new_work, start_time, end_time, new_worker))

        # Update transmission statistics.
        stat_row = self.df_worker_stats.loc[new_worker]
        stat_row["Weight transmissions"] += new_worker != weight_worker
        stat_row["Activation transmissions"] += new_worker != pred_worker
        stat_row["Activation storage"] += new_work.direction.value
        stat_row["Max activation storage"] = max(
            stat_row["Max activation storage"], stat_row["Activation storage"]
        )

        if new_worker == weight_worker:
            self.worker_weight_storage[new_worker].add(new_work.stage)

    def max_time(self) -> Timestamp:
        return max(work_history.end_time for work_history in self.all_work)

    def render_work_produced(self) -> str:
        # Populate a 2D array that represents the pipeline diagram.
        pipeline_diagram = np.full(
            (self.num_workers, self.max_time()), "<td colspan=1></td>", object
        )
        for work_history in self.all_work:
            length = work_history.end_time - work_history.start_time
            pipeline_diagram[work_history.performed_by, work_history.start_time] = (
                f"<td class='batch{work_history.work.batch}' colspan='{length}'>{work_history.work}</td>"
            )
            pipeline_diagram[
                work_history.performed_by,
                work_history.start_time + 1 : work_history.end_time,
            ] = ""

        # Render the pipeline diagram as an HTML table.
        max_char_width = 2 + max(
            len(str(work_history.work)) for work_history in self.all_work
        )
        num_batches = 1 + max(work_history.work.batch for work_history in self.all_work)
        batch_styles = ""
        for batch in range(num_batches):
            color = np.array(matplotlib.cm.Set1(batch / num_batches)[:-1])
            batch_styles += f"""
            .timing .batch{batch} {{
                background-color: {matplotlib.colors.to_hex(color)};
                border: 1px solid {matplotlib.colors.to_hex(color * 0.9)};
            }}
            """
        time_header = "".join(f"<th>{t+1}</th>" for t in range(self.max_time()))
        content = "\n".join(
            f"<tr><td>w{worker}</td>" + "".join(pipeline_diagram_row) + "</tr>"
            for worker, pipeline_diagram_row in enumerate(pipeline_diagram)
        )
        return display.HTML(f"""
            <head>
                <style>
                    .timing {{ border-spacing: 0; }}
                    .timing td {{
                        padding: 0px;
                        text-align: center;
                        color: #888;
                        min-width: {max_char_width}ch;
                    }}
                    .timing th {{
                        font-weight: normal;
                        font-size: 9px;
                        width: 8ch;
                        text-align: right;
                    }}
                    .timing td:first-child {{
                        border: 0px;
                        font-weight: normal;
                        padding-right: 8px;
                    }}
                    .timing header {{
                        border: 0px
                    }}
                    .timing .schedule {{
                        text-align: center;
                        font-size: 12px;
                    }}
                    {batch_styles}
                </style>
            </head>

            <body>
            <table class='timing' align='center'>
                <tr><th colspan=100% class='schedule'>{self.schedule.__name__}</th></tr>
                <tr><th></th>{time_header}</tr>
                {content}
            </table>
            </body>
        """)

    def save_work_produced(self, filename: str) -> "SimulationStats":
        with open(filename, "w") as f:
            f.write(self.render_work_produced()._repr_html_())
        return self

    def worker_stats(self) -> pd.DataFrame:
        df = self.df_worker_stats.copy()

        # Delete 'Activation storage'. It'll just be 0. We care about peak
        # activation storage.
        assert (
            df["Activation storage"] == 0
        ).all(), "There is a bug in the simulator. Every forward must have a matching backward."
        del df["Activation storage"]

        # Populate 'Weight storage' by counting the number of stages for each worker
        # where w_compute was = w_storage.
        for worker, stages in self.worker_weight_storage.items():
            df.loc[worker, "Weight storage"] = len(stages)

        return df

    def _repr_html_(self):
        return (
            self.render_work_produced()._repr_html_()
            + "<br>\n"
            + self.worker_stats()._repr_html_()
        )


def simulate(
    num_workers: int,
    num_stages: int,
    num_batches: int,
    schedule: Callable[[Stage, Batch], ComputeAndWeightWorkers],
    priority_order: Callable[[Stage, Batch, Direction], int] = None,
) -> SimulationStats:
    if priority_order is None:
        priority_order = BackwardFirst(num_stages, num_batches)
    stats = SimulationStats(num_workers, schedule)

    execution_frontier: Set[Work] = set(
        Work(-1, batch, Direction.FORWARD) for batch in range(num_batches)
    )
    work_start_time: Dict[Work, Timestamp] = {work: 0 for work in execution_frontier}
    work_end_time: Dict[Work, Timestamp] = {work: 0 for work in execution_frontier}
    worker_busy_until: Dict[Worker, Timestamp] = collections.defaultdict(lambda: 0)

    while execution_frontier:
        r: List[Tuple[Timestamp, Work, Work, Worker]] = []
        for work in execution_frontier:
            new_work = work.next(num_stages)
            worker = schedule(new_work).compute_worker
            when_work_can_start = max(worker_busy_until[worker], work_end_time[work])
            priority = priority_order(work)
            r.append((when_work_can_start, priority, work, new_work, worker))

        when_work_can_start, priority, old_work, new_work, new_worker = min(r)

        execution_frontier.remove(old_work)
        execution_frontier.add(new_work)
        work_start_time[new_work] = when_work_can_start
        work_end_time[new_work] = worker_busy_until[new_worker] = (
            when_work_can_start + new_work.duration()
        )

        stats.add(
            old_work, new_work, work_start_time[new_work], work_end_time[new_work]
        )

        # Remove finished work from the frontier.
        execution_frontier = {
            work for work in execution_frontier if not work.is_final()
        }

    return stats
