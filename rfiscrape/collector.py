"""A microservice for collecting the RFI data and writing into a buffer."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from aiohttp import web

from . import config, db, prioritymap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s:%(message)s",
)
logging.getLogger("aiohttp").setLevel(logging.WARN)
logger = logging.getLogger(__name__)


# A type for storing the pushed data in the queue
@dataclass(order=True)
class PrioritizedItem:
    """A simple wrapper for storing prioritized data in the queue."""

    priority: int
    item: dict = field(compare=False)


# A type for storing the assembled data
class AssembledData:
    """A simple container for the assembled data."""

    def __init__(self, timestamp: float, nfreq: int):
        self.timestamp = timestamp

        self.dropped = np.full((2, nfreq), fill_value=np.nan, dtype=np.float32)
        self.total = np.full((2, nfreq), fill_value=np.nan, dtype=np.float32)


assemble_queue = queue.PriorityQueue()
write_queue = queue.Queue()


# A very simple async receiver, just push the data into a queue for another thread to
# consume
async def receive_rfi(request: web.Request) -> web.Response:
    """Handler for receiving RFI stats from the clients."""
    body = await request.json()

    assemble_queue.put(PrioritizedItem(priority=body["time"], item=body))

    return web.Response()


def assembler(window: int, nfreq: int) -> None:
    """A worker to take the entries for each time/freq combo and assemble into samples.

    Parameters
    ----------
    window
        The number of time samples to keep open.
    nfreq
        The number of frequencies we are expecting.
    """
    entries = prioritymap.PriorityMap(window, strict=True, ignore_existing=True)

    def _create(timestamp: float):
        def f():
            return AssembledData(timestamp, nfreq)

        return f

    while True:
        v = assemble_queue.get()
        assemble_queue.task_done()

        # If we get None that is the signal we should cleanup and exit, so we break from
        # this loop
        if v is None:
            break

        data = v.item

        timestamp = data["time"]

        # Successful completion of this block should ensure that an entry for this
        # timestamp exists
        try:
            oldkey, oldvalue = entries.pushpop(timestamp, call=_create(timestamp))
        except prioritymap.OutOfOrderError:
            oldest = entries.peek()[0]
            logger.info(
                f"Received an entry with too old a timestamp {timestamp}. "
                f"Current oldest {oldest}",
            )
            continue

        # We pushed out an old value, so we need to send it on for the next processing
        # stage
        if oldkey is not None:
            write_queue.put(oldvalue)

        # Update the entry fot this timestamp
        entry = entries[timestamp]
        freq_ind = np.array(data["freq"])

        if (freq_ind >= nfreq).any() or (freq_ind < 0).any():
            raise ValueError("Frequency indices out of range.")

        counts = np.array(data["data"]).astype(np.float32)

        try:
            entry.dropped[:, freq_ind] = counts[:, 1]
            entry.total[:, freq_ind] = counts[:, 0]
        except IndexError:
            logger.exception(f"Issue with indexing skipping. {freq_ind=}")

    # We need to pass all the current entries along to get written out
    while len(entries) > 0:
        write_queue.put(entries.pop()[1])

    # Send a None to the writing stages to have them exit
    write_queue.put(None)

    # Wait until the writer exits
    write_queue.join()


def writer(
    output_file: str, buffer_time: float, purge_interval: float,
) -> None:
    """Write out the data into a sqlite buffer.

    This will store the actual amount of flagged samples within each interval by
    differencing between samples.

    Parameters
    ----------
    output_file
        The name of the output file.
    buffer_time
        The length of the buffer to maintain.
    purge_interval
        How often to remove expired samples in seconds.
    """
    db.connect(output_file, readonly=False)

    prev = None

    # Set the last_purge to be at the epoch to ensure we purge at the first run, and
    # determine how often to purge.
    last_purge = 0.0

    while True:
        item = write_queue.get()
        write_queue.task_done()

        # None indicates we won't get any more data, so we just exit the loop
        if item is None:
            break

        data: AssembledData = item

        if prev is None:
            prev = data
            continue

        # Calculate the amount of dropped samples between the current and previous
        # samples, and the reset the `prev` ref
        dropped_frac = (data.dropped - prev.dropped) / (data.total - prev.total)
        prev = data

        # Log some output
        avg_excision = np.nanmean(dropped_frac, axis=-1)
        logger.info(
            f"Writing {data.timestamp}. Mean excision fraction: "
            f"stage1={avg_excision[0]:.3%} stage2={avg_excision[1]:.3%}",
        )

        db.RFIData.create(
            timestamp=data.timestamp,
            freq_chunk=0,
            spectrum_type=db.SpectrumType.STAGE_1,
            encoding_type=db.EncodingType.RAW,
            data=dropped_frac[0].data,
        )
        db.RFIData.create(
            timestamp=data.timestamp,
            freq_chunk=0,
            spectrum_type=db.SpectrumType.STAGE_2,
            encoding_type=db.EncodingType.RAW,
            data=dropped_frac[1].data,
        )

        # Check how long it's been since we purged old record, and if it's been too long
        # DELETE anything older than the buffer_time
        if (time.time() - last_purge) > purge_interval:
            oldest_record = time.time() - buffer_time
            delete_query = db.RFIData.delete().where(
                db.RFIData.timestamp < oldest_record,
            )

            records_deleted = delete_query.execute()

            logger.info(f"Purged {records_deleted} expired records.")
            last_purge = time.time()

    # Close the database to allow in all to get synced
    db.close()


def main() -> None:
    """The CLI entrypoint."""
    # Parse the command line arguments
    conf = config.process_args_and_config(
        prog="rfiscrape-collector",
        description="Collect RFI stats from kotekan and assemble into a ondisk buffer.",
        schema=config.schema,
        sections=["common", "server"],
        configbase="rfiscrape",
    )
    nfreq = 1024

    # Convert the purge config into an actual interval in seconds
    if conf["purge"] <= 0:
        raise ValueError("Purge interval must be positive.")
    purge = conf["time"] * conf["purge"] if conf["purge"] < 1 else conf["purge"]

    # Start the assembly and writing threads
    assembler_thread = threading.Thread(target=assembler, args=(conf["window"], nfreq))
    writer_thread = threading.Thread(
        target=writer, args=(conf["buffer"], conf["time"], purge),
    )
    assembler_thread.start()
    writer_thread.start()

    # Start the collector HTTP server on the main thread
    app = web.Application()
    app.add_routes([web.post("/rfi", receive_rfi)])
    web.run_app(app, port=conf["port"])

    # If this exits then we need to flush and close the threads. We do this by placing
    # the None sentinel into the queue which signals a shutdown
    logger.info("Shutting down threads.")
    assemble_queue.put(None)
    assembler_thread.join()
    writer_thread.join()
    logger.info("Exiting.")


if __name__ == "__main__":
    main()
