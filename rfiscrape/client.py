"""A minimal dependency script for fetching kotekan RFI data and sending it on."""

import argparse
import math
import random
import re
import time
from email.utils import parsedate_to_datetime

import requests

# Output JSON format
# schema = {
#     "time": UTC,
#     "freq_id": [ids],
#     "data": [[[ 3Darray ]]],
# }

# Regex for selecting the relevant metrics and extracting the required parts
metric_pattern = re.compile(
    r"^kotekan_rfi(?P<stage>broadcast|framedrop)_?(?P<type>\w*)_"
    r"(?:sample|frame)_total\{.*freq_id=\"(?P<freq>\d+)\"\} (?P<count>\d+)",
)


# Mapping for the locations of the entries we want in the output
stage_ind = {"broadcast": 0, "framedrop": 1}
type_ind = {"total": 0, "dropped": 1}


def scrape(target: str, timeout: float) -> dict:
    """Scrape the current RFI data from the target.

    Timeout after the given number of seconds.
    """
    r = requests.get(f"http://{target}/metrics", timeout=timeout)

    metrics = r.content.decode().splitlines()

    matched_metrics = []

    freq = set()

    # First pass over the scraped metrics.
    # Scrape the ones we want, massage the contents and determine which frequencies are
    # present
    for m in metrics:
        mo = re.match(metric_pattern, m)

        if mo is not None:
            params = mo.groupdict()
            params["freq"] = int(params["freq"])
            params["count"] = int(params["count"])
            params["type"] = params["type"] or "total"
            freq.add(params["freq"])
            matched_metrics.append(params)

    # Determine the frequency ordering
    freq_ind = {f: ii for ii, f in enumerate(sorted(freq))}

    # The output array. We need to explicitly create the entries to avoid shallow copies
    output = [[[-1] * len(freq_ind) for _ in range(2)] for _ in range(2)]

    # Second pass over the metrics, copy the count over into the correct locations
    for m in matched_metrics:
        try:
            si = stage_ind[m["stage"]]
            ti = type_ind[m["type"]]
            fi = freq_ind[m["freq"]]
        except KeyError:
            # Unknown metric
            continue

        output[si][ti][fi] = m["count"]

    timestamp = parsedate_to_datetime(r.headers["Date"]).timestamp()

    return {
        "time": timestamp,
        "freq": sorted(freq),
        "data": output,
    }


def main() -> None:
    """Client main loop."""
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        prog="rfiscrape-client",
        description="Extract local RFI stats from prometheus and send to a collector",
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        help="Wait time in seconds between scrapes.",
        default=5,
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Scrape target given as hostname:port. Defaults to localost:12048",
    )
    parser.add_argument("server", type=str, help="The hostname:port of the collector.")
    args = parser.parse_args()

    scrape_timeout = args.interval / 4
    push_timeout = args.interval / 4
    server_url = f"http://{args.server}/rfi"

    session = requests.Session()

    while True:
        # Calculate the next target time, this should be an exact interval boundary
        next_time = args.interval * int(math.ceil(time.time() / args.interval))

        # Wait until the next sequence start
        time.sleep(next_time - time.time())

        # Fetch the data and insert the sequence ID
        try:
            s = scrape(args.target, scrape_timeout)
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout):
            continue

        # Insert the target time
        s["completion_time"] = s["time"]
        s["time"] = next_time

        # Sleep a random amount to spread out the requests on the server
        time_until_next = next_time + args.interval - time.time()
        random_sleep = 0.2 * time_until_next * random.random()  # noqa: S311
        time.sleep(random_sleep)

        # Push over to the collection server
        try:
            session.post(server_url, json=s, timeout=push_timeout)
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout):
            continue


if __name__ == "__main__":
    main()
