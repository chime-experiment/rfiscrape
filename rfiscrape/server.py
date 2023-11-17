"""HTTP server for accessing RFI stats from the buffer."""
import argparse

import numpy as np
from aiohttp import web

from . import db, util


# A very simple async receiver, just push the data into a queue for another thread to
# consume
async def get_rfi(request: web.Request) -> web.Response:
    """Handler for RFI data requests."""
    body = await request.json()
    try:
        start_time = util.convert_unix(body["start_time"])
        end_time = util.convert_unix(body["end_time"])
    except KeyError as e:
        raise RuntimeError("Start and end times are required.") from e
    except RuntimeError as e:
        raise RuntimeError("Error parsing times.") from e

    spec_type = body.get("spectrum_type", db.SpectrumType.STAGE_1)

    freq_start = body.get("freq_start", None)
    freq_end = body.get("freq_end", None)

    timestamps, freq, data = query_db(
        start_time, end_time, spec_type, freq_start, freq_end,
    )

    encoded_data = {
        "time": util.numpy_to_json(timestamps),
        "freq": util.numpy_to_json(freq),
        "data": util.numpy_to_json(data),
    }

    return web.json_response(encoded_data)


def decode(
    spec: db.SpectrumType, enc: db.EncodingType, data: bytes,
) -> np.ndarray:
    """Decode the data in the RFIData result."""
    return np.frombuffer(data, dtype=np.float32, count=-1)


def query_db(
    start_time: float,
    end_time: float,
    spec_type: db.SpectrumType,
    freq_start: int | None = None,
    freq_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read RFI information from the buffer db."""
    query = db.RFIData.select().where(db.RFIData.spectrum_type == spec_type)
    query = query.where(
        db.RFIData.timestamp > start_time, db.RFIData.timestamp < end_time,
    )

    # Add the frequency constraints if set
    # if freq_start:
    #     query = query.where(db.RFIData.freq_low < freq_end)
    # if freq_end:
    #     query = query.where(db.RFIData.freq_high > freq_start)

    results = list(query)

    # Get the list of timestamps. Use a set as they may be multiple frequency chunks
    timestamps = np.array(sorted({r.timestamp for r in results}))

    ts_map = {ts: ii for ii, ts in enumerate(timestamps)}

    chunks = sorted({r.freq_chunk for r in results})  # This should by [0]
    chunksize = 1024  # TODO: look this up from the spectrum/encoding type
    freq = np.arange(chunks[0] * chunksize, (chunks[-1] + 1) * chunksize)

    output_data = np.full(
        (len(timestamps), len(chunks), chunksize), fill_value=np.nan, dtype=np.float32,
    )

    for r in results:
        ti = ts_map[r.timestamp]
        ci = 0

        output_data[ti, ci] = decode(spec_type, r.encoding_type, r.data)

    # Flatten across chunks
    output_data = output_data.reshape(len(timestamps), -1)

    # Extract the required part
    if freq_start is not None or freq_end is not None:
        output_data = output_data[:, freq_start:freq_end].copy()
        freq = freq[freq_start:freq_end]

    return timestamps, freq, output_data


def main() -> None:
    """Main entry point for the RFI data server."""
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        prog="rfiscrape-server",
        description="Serve the data from the buffer over HTTP.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Name of the buffer output file",
        default="buffer.sql",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        help="Length of the buffer in seconds.",
        default=60,
    )
    parser.add_argument(
        "--purge",
        type=int,
        help=(
            "Interval between purging expired samples. "
            "If not set, uses 10% of the buffer length."
        ),
        default=None,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen on.",
        default=8464,
    )
    nfreq = 1024
    parser.add_argument("server", type=str, help="The hostname:port of the collector.")
    args = parser.parse_args()
if __name__ == "__main__":

    db.connect("buffer.sql", readonly=True)

    app = web.Application()
    app.add_routes([web.get("/query", get_rfi)])
    web.run_app(app, port=8465)

    db.close()
