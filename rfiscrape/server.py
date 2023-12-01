"""HTTP server for accessing RFI stats from the buffer."""
from io import BytesIO

import h5py
import numpy as np
from aiohttp import web

from . import config, db, util


def _rfidata_to_json(time: np.ndarray, freq: np.ndarray, data: np.ndarray) -> dict:
    return {
        "time": util.numpy_to_json(time),
        "freq": util.numpy_to_json(freq),
        "data": util.numpy_to_json(data),
    }


def _rfidata_to_numpy_bytes(
    time: np.ndarray, freq: np.ndarray, data: np.ndarray,
) -> dict:
    with BytesIO() as f:
        np.savez(f, time=time, freq=freq, data=data)
        return f.getvalue()


def _rfidata_to_h5py_bytes(
    time: np.ndarray, freq: np.ndarray, data: np.ndarray,
) -> dict:
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
    except KeyError as e:
        raise web.HTTPBadRequest(reason="Start and end times are required.") from e
    except RuntimeError as e:
        raise web.HTTPBadRequest(reason="Error parsing times.") from e

    freq_start = query.get("freq_start", None)
    freq_end = query.get("freq_end", None)
    type_ = query.get("type", "json")

    spec_type = query.get("spectrum_type", "STAGE_1")
    try:
        spec_type = db.SpectrumType[spec_type]
    except KeyError as e:
        raise web.HTTPBadRequest(reason=f"Unknown spectrum type {spec_type=}.") from e

    rfi_data = db.fetch_rfi(
        start_time,
        end_time,
        spec_type,
        freq_start,
        freq_end,
    )

    match type_:
        case "json":
            d = _rfidata_to_json(*rfi_data)
            r = web.json_response(d)
        case "numpy":
            r = web.Response()
            r.headers["Content-Disposition"] = "Attachment;filename=rfi.npz"
            r.headers["Content-Type"] = "application/x-python"
            r.body = _rfidata_to_numpy_bytes(*rfi_data)
        case "hdf5":
            r = web.Response()
            r.headers["Content-Disposition"] = "Attachment;filename=rfi.h5"
            r.headers["Content-Type"] = "application/x-hdf5"
            r.body = _rfidata_to_h5py_bytes(*rfi_data)
        case _:
            raise web.HTTPBadRequest(reason=f"Unknown type {type_}.")

    return r


def main() -> None:
    """Main entry point for the RFI data server."""
    # Parse the command line arguments
    conf = config.process_args_and_config(
        "rfiscrape-server",
        "Serve the data from the buffer over HTTP.",
        config.schema,
        ["common", "server"],
    )

    db.connect(conf["buffer"], readonly=True)

    app = web.Application()
    app.add_routes([web.get("/query", get_rfi)])
    web.run_app(app, port=conf["port"])

    db.close()
