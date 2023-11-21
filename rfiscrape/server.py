"""HTTP server for accessing RFI stats from the buffer."""
import argparse
from io import BytesIO

from aiohttp import web
import h5py
import numpy as np

from . import db, util


def rfidata_to_json(time: np.ndarray, freq: np.ndarray, data: np.ndarray) -> dict:
    return {
        "time": util.numpy_to_json(time),
        "freq": util.numpy_to_json(freq),
        "data": util.numpy_to_json(data),
    }

def rfidata_to_numpy_bytes(time: np.ndarray, freq: np.ndarray, data: np.ndarray) -> dict:

    with BytesIO() as f:
        np.savez(f, time=time, freq=freq, data=data)
        return f.getvalue()


def rfidata_to_h5py_bytes(time: np.ndarray, freq: np.ndarray, data: np.ndarray) -> dict:

    with BytesIO() as f:
        with h5py.File(f, mode="w") as fh:
            fh.create_dataset("index_map/time", data=time)
            fh.create_dataset("index_map/freq", data=freq)
            ds = fh.create_dataset("rfi", data=data)
            ds.attrs["axis"] = ["time", "freq"]
        return f.getvalue()


async def get_rfi(request: web.Request) -> web.Response:
    """Handler for RFI data requests."""

    query_uri = request.query
    query_json = await request.json() if request.body_exists else None

    if query_uri and query_json:
        raise web.HTTPBadRequest(
            reason="Use either a query string, or a JSON body, not both.",
        )

    query = query_uri or query_json

    if not query:
        raise web.HTTPBadRequest(
            reason="Supply a query, either as a string, or a JSON body.",
        )

    try:
        start_time = util.convert_unix(query["start_time"])
        end_time = util.convert_unix(query["end_time"])
    except KeyError:
        raise web.HTTPBadRequest(reason="Start and end times are required.")
    except RuntimeError:
        raise web.HTTPBadRequest(reason="Error parsing times.")

    spec_type = query.get("spectrum_type", db.SpectrumType.STAGE_1)
    freq_start = query.get("freq_start", None)
    freq_end = query.get("freq_end", None)
    type = query.get("type", "json")

    rfi_data = db.fetch_rfi(
        start_time, end_time, spec_type, freq_start, freq_end,
    )

    if type == "json":
        d = rfidata_to_json(*rfi_data)
        r = web.json_response(d)
    elif type == "numpy":
        r = web.Response()
        r.headers["Content-Disposition"] = "Attachment;filename=rfi.npz"
        r.headers["Content-Type"] = "application/x-python"
        r.body = rfidata_to_numpy_bytes(*rfi_data)
    elif type == "hdf5":
        r = web.Response()
        r.headers["Content-Disposition"] = "Attachment;filename=rfi.h5"
        r.headers["Content-Type"] = "application/x-hdf5"
        r.body = rfidata_to_h5py_bytes(*rfi_data)
    else:
        raise web.HTTPBadRequest(reason=f"Unknown type {type}.")

    return r


def main() -> None:
    """Main entry point for the RFI data server."""
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        prog="rfiscrape-server",
        description="Serve the data from the buffer over HTTP.",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=str,
        help="Name of the buffer file. Defaults to 'buffer.sql'.",
        default="buffer.sql",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen on. Defaults to 8465.",
        default=8465,
    )
    args = parser.parse_args()

    db.connect(args.buffer, readonly=True)

    app = web.Application()
    app.add_routes([web.get("/query", get_rfi)])
    web.run_app(app, port=args.port)

    db.close()
