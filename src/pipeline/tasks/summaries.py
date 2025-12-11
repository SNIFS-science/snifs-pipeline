import json
import sqlite3
from datetime import datetime as dt
from pathlib import Path

import numpy as np

from pipeline.common.image import Image
from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.resolver.common import FileStoreEntry
from pipeline.resolver.resolver import Resolver


@pipeline_task()
def summarise_image(
    image: Image, file: FileStoreEntry, output_location: Path, discriminator: str
) -> dict[str, str | int | float | dt]:
    file_store_data = file.model_dump(exclude_none=True)
    good_pixels = np.isfinite(image.data) & np.isfinite(image.variance)
    if "NUM_BAD_PIXELS" not in image.header:
        num_bad_pixels = int(np.sum(~good_pixels, axis=None))
        image.header.set("NUM_BAD_PIXELS", num_bad_pixels, metric=True)
    output_location.parent.mkdir(parents=True, exist_ok=True)
    prefect_context = {}
    try:
        from prefect.context import TaskRunContext, get_run_context

        context: TaskRunContext = get_run_context()  # type: ignore
        prefect_context = {
            "flow_run_id": str(context.task_run.flow_run_id),
            "task_run_id": str(context.task_run.id),
            "api_url": str(context.client.api_url),
        }
        start_time = context.task_run.start_time
        if start_time is not None:
            prefect_context["task_start_time"] = start_time.isoformat()

    except Exception:
        pass
    content = {
        **{k: v for k, v in file_store_data.items() if v is not None and v != ""},
        **image.header.get_metrics(lowercase=True),
        **prefect_context,
        "discriminator": discriminator,
        "public_dir": str(output_location.parent),
    }

    to_write = json.dumps(content, indent=2, default=lambda x: x.isoformat() if isinstance(x, dt) else str(x))
    logger = get_logger()
    logger.info(f"Writing summary to {output_location}")
    logger.debug(f"Summary content:\n{to_write}")
    output_location.write_text(to_write)
    return content


@pipeline_task()
def write_summary(resolver: Resolver, content: dict[str, str | int | float | dt]) -> None:
    with sqlite3.connect(resolver.database_path) as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN TRANSACTION;")
        cursor.execute(
            """
            INSERT OR REPLACE INTO summaries (task_run_id, flow_run_id, discriminator, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (content["task_run_id"], content["flow_run_id"], content["discriminator"], dt.now()),
        )
        keys = set(content.keys()) - {"task_run_id", "flow_run_id", "discriminator"}
        for key in keys:
            val = content[key]
            if isinstance(val, str):
                value_str = val
                value_num = None
            elif isinstance(val, dt):
                value_str = val.isoformat()
                value_num = val.timestamp()
            elif isinstance(val, (int, float)):
                value_str = str(val)
                value_num = val
            elif isinstance(val, list):
                value_str = json.dumps(val)
                value_num = None
            elif isinstance(val, bool):
                value_str = str(val).lower()
                value_num = 1 if val else 0
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(val)}")
            cursor.execute(
                """
                INSERT OR REPLACE INTO summary_info (task_run_id, key, value_str, value_num)
                VALUES (?, ?, ?, ?)
                """,
                (content["task_run_id"], key, value_str, value_num),
            )
        cursor.execute("COMMIT;")
        conn.commit()
