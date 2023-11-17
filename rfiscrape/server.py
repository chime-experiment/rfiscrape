"""HTTP server for accessing RFI stats from the buffer."""
import argparse

from aiohttp import web

from . import db, util


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

    timestamps, freq, data = db.fetch_rfi(
        start_time, end_time, spec_type, freq_start, freq_end,
    )

    encoded_data = {
        "time": util.numpy_to_json(timestamps),
        "freq": util.numpy_to_json(freq),
        "data": util.numpy_to_json(data),
    }

    return web.json_response(encoded_data)


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
