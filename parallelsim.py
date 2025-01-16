from typing import Callable, Dict, List, NamedTuple, Set, Tuple

import collections
import dataclasses as dc
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

    def successor(self, num_stages: int) -> "Work":
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


@dc.dataclass(frozen=True)
class PriorityOrder:
    order: Dict[Work, int]

    def __call__(self, work: Work) -> int:
        return self.order[work]


class OldestStageFirst(PriorityOrder):
    def __init__(self, num_stages: int, num_batches: int):
        super().__init__(
            {
                Work(stage, batch, direction): n
                for n, (stage, direction, batch) in enumerate(
                    itertools.product(
                        range(-1, num_stages), Direction, range(num_batches)
                    )
                )
            }
        )


class OldestBatchFirst(PriorityOrder):
    def __init__(self, num_stages: int, num_batches: int):
        super().__init__(
            {
                Work(stage, batch, direction): n
                for n, (batch, direction, stage) in enumerate(
                    itertools.product(
                        range(num_batches), Direction, range(-1, num_stages)
                    )
                )
            }
        )


class WorkHistory(NamedTuple):
    work: Work
    start_time: Timestamp
    end_time: Timestamp
    performed_by: Worker


@dc.dataclass
class SimulationStats:
    num_workers: int
    num_stages: int
    num_batches: int
    schedule: Callable[[Work], ComputeAndWeightWorkers]

    def __post_init__(self) -> None:
        self.all_work: List[WorkHistory] = []
        self.df_worker_stats = pd.DataFrame(
            0,
            index=range(self.num_workers),
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

    def update(
        self,
        frontier_work: Work,
        new_work: Work,
        start_time: Timestamp,
        end_time: Timestamp,
    ) -> None:
        frontier_worker = self.schedule(frontier_work).compute_worker
        new_worker, weight_worker = self.schedule(new_work)

        # Record the work.
        self.all_work.append(WorkHistory(new_work, start_time, end_time, new_worker))

        worker_stats = self.df_worker_stats.loc[new_worker]

        # This work doesn't own its weights. Grab them from the owner.
        worker_stats["Weight transmissions"] += new_worker != weight_worker

        # Grab the activations from the predecessor.
        worker_stats["Activation transmissions"] += new_worker != frontier_worker

        if new_work.direction == Direction.FORWARD:
            backward_worker = self.schedule(
                Work(new_work.stage, new_work.batch, Direction.BACKWARD)
            ).compute_worker
            if new_worker == backward_worker:
                # This worker stores its input until it computes the backward
                # pass.
                worker_stats["Activation storage"] += 1
            else:
                # This worker transmits its input to backward_worker, which stores
                # it until it computes the backward pass.
                worker_stats["Activation transmissions"] += 1
                self.df_worker_stats.loc[backward_worker, "Activation storage"] += 1
        else:
            # As soon as the backward pass finishes, the worker can free its
            # input.
            worker_stats["Activation storage"] -= 1

        worker_stats["Max activation storage"] = max(
            worker_stats["Max activation storage"], worker_stats["Activation storage"]
        )

        # Collect weights owned by this worker.
        if new_worker == weight_worker:
            self.worker_weight_storage[new_worker].add(new_work.stage)

    def end_time(self) -> Timestamp:
        return max(work_history.end_time for work_history in self.all_work)

    def render_work_produced(self) -> display.HTML:
        # Populate a 2D array that represents the pipeline diagram.
        pipeline_diagram = np.full(
            (self.num_workers, self.end_time()), "<td colspan=1></td>", object
        )
        for wh in self.all_work:
            length = wh.end_time - wh.start_time
            pipeline_diagram[wh.performed_by, wh.start_time] = (
                f"<td class='batch{wh.work.batch}' colspan='{length}'>{wh.work}</td>"
            )
            pipeline_diagram[wh.performed_by, wh.start_time + 1 : wh.end_time] = ""

        # Render the pipeline diagram as an HTML table.
        max_char_width = 2 + max(len(str(wh.work)) for wh in self.all_work)
        batch_styles = "\n".join(
            f"""
            .timing .batch{batch} {{
                background-color: {matplotlib.colors.to_hex(color)};
                border: 1px solid {matplotlib.colors.to_hex(color * 0.9)};
            }}
            """
            for batch, color in enumerate(
                matplotlib.cm.tab20(np.linspace(0, 1, self.num_batches))[:, :-1]
            )
        )
        timing_table_header = "".join(
            f"<th>{t + 1}</th>" for t in range(self.end_time())
        )
        timing_table_rows = "\n".join(
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
                <tr><th></th>{timing_table_header}</tr>
                {timing_table_rows}
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
        assert (df["Activation storage"] == 0).all(), (
            "There is a bug in the simulator. Every forward must have a matching backward."
        )
        del df["Activation storage"]

        # Populate 'Weight storage' by counting the number of stages for each worker
        # where w_compute was = w_storage.
        for worker, stages in self.worker_weight_storage.items():
            df.loc[worker, "Weight storage"] = len(stages)

        return df

    def aggregate_stats(self) -> pd.DataFrame:
        df_worker_stats = self.worker_stats()
        return pd.DataFrame(
            pd.Series(
                {
                    "Worker throughput (jobs / time / worker)": len(self.all_work)
                    / self.end_time()
                    / self.num_workers,
                    "Max activation storage for a worker": df_worker_stats[
                        "Max activation storage"
                    ].max(),
                    "Mean activation storage for workers": df_worker_stats[
                        "Max activation storage"
                    ].mean(),
                },
                name="Aggegrate Metrics",
            )
        )

    def _repr_html_(self):
        return (
            self.aggregate_stats()._repr_html_()
            + "<br>\n"
            + self.render_work_produced()._repr_html_()
            + "<br>\n"
            + self.worker_stats()._repr_html_()
        )


def simulate(
    num_workers: int,
    num_stages: int,
    num_batches: int,
    schedule: Callable[[Work], ComputeAndWeightWorkers],
    priority_order: Callable[[Work], int] = None,
) -> SimulationStats:
    # Simulate a set of workers that greedily grab the next job from a DAG of
    # jobs. A pleasant aspect of this simulation is that it's tickless, and it
    # doesn't doesn't maintain a global clock. Instead of monotonically
    # advancing a clock at each round of simulation, it finds the work that can
    # be executed the earliest, and deduces time from it. This simulates the
    # asynchronous behavior of the workers more closely than a tick-based
    # simulator.

    priority_order = priority_order or OldestBatchFirst(num_stages, num_batches)
    stats = SimulationStats(num_workers, num_stages, num_batches, schedule)

    # The execution frontier is the set of work that has already executed,
    # but which have descendents that have not yet executed.
    execution_frontier: Set[Work] = set(
        Work(-1, batch, Direction.FORWARD) for batch in range(num_batches)
    )
    # For work that has already been simulated, these dictionaries track when
    # the work started and ended.
    work_start_time: Dict[Work, Timestamp] = {work: 0 for work in execution_frontier}
    work_end_time: Dict[Work, Timestamp] = {work: 0 for work in execution_frontier}

    # When each worker becomes free next.
    worker_busy_until: Dict[Worker, Timestamp] = collections.defaultdict(lambda: 0)

    while execution_frontier:
        # Find work that can execute the earliest.
        r: List[Tuple[Timestamp, Work, Work, Worker]] = []
        for frontier_work in execution_frontier:
            new_work = frontier_work.successor(num_stages)
            worker = schedule(new_work).compute_worker
            when_work_can_start = max(
                worker_busy_until[worker], work_end_time[frontier_work]
            )
            priority = priority_order(frontier_work)
            r.append((when_work_can_start, priority, frontier_work, new_work, worker))
        when_work_can_start, priority, frontier_work, new_work, new_worker = min(r)

        # Replace the work work with its descendent.
        execution_frontier.remove(frontier_work)
        execution_frontier.add(new_work)

        work_start_time[new_work] = when_work_can_start
        work_end_time[new_work] = worker_busy_until[new_worker] = (
            when_work_can_start + new_work.duration()
        )

        # Remove finished work from the frontier.
        execution_frontier = {
            work for work in execution_frontier if not work.is_final()
        }

        stats.update(
            frontier_work, new_work, work_start_time[new_work], work_end_time[new_work]
        )

    return stats
