"""HTTP server for accessing RFI stats from the buffer."""
import datetime
from pathlib import Path

import h5py
from peewee import fn

from . import config, db


def archive(
    filename: str,
    dtstart: datetime.datetime,
    dtend: datetime.datetime,
    types: list[db.SpectrumType] | None = None,
) -> None:
    """Archive a span of data into an HDF5 file."""
    data = []

    if types is None:
        types = [db.SpectrumType.STAGE_1, db.SpectrumType.STAGE_2]

    for spec_type in types:
        res = db.fetch_rfi(
            dtstart.timestamp(),
            dtend.timestamp(),
            spec_type,
        )
        data.append(res)

    time0 = data[0][0]
    freq0 = data[0][1]

    for t, f, _ in data[1:]:
        if (t != time0).any() or (f != freq0).any():
            raise RuntimeError("Can only spectrum types with a single time axis.")

    with h5py.File(filename, mode="w") as fh:
        fh.create_dataset("index_map/time", data=time0)
        fh.create_dataset("index_map/freq", data=freq0)

        for st, (_, __, d) in zip(types, data):
            ds = fh.create_dataset(f"rfi_{st.name.lower()}", data=d)
            ds.attrs["axis"] = ["time", "freq"]


def main() -> None:
    """Main entry point for the RFI data server."""
    conf = config.process_args_and_config(
        prog="rfiscrape-archiver",
        description="Archive data into HDF5 files.",
        schema=config.schema,
        sections=["common", "archiver"],
        configbase="rfiscrape",
        extra_args=[("--date", {"type": str, "default": None})],
    )

    spectrum_types = []

    try:
        spectrum_types = [db.SpectrumType[st] for st in conf["types"]]
    except KeyError as e:
        raise RuntimeError("Unknown spectrum type.") from e

    db.connect(conf["buffer"], readonly=True)

    # Archive a single file
    if conf["date"] is not None:
        try:
            dtstart = datetime.datetime.fromisoformat(conf["date"]).replace(
                tzinfo=datetime.timezone.utc,
            )
        except ValueError as e:
            raise ValueError(f"Did not understand date {conf['date']}") from e

        dtend = dtstart + datetime.timedelta(days=1)

        archive(
            f"rfi_{dtstart.strftime('%Y-%m-%d')}.h5", dtstart, dtend, spectrum_types
        )
        return

    # Otherwise try and process all missing files.

    # Get the range of times in the ringbuffer as datetimes
    earliest, latest = db.RFIData.select(
        fn.Min(db.RFIData.timestamp), fn.Max(db.RFIData.timestamp)
    ).scalar(as_tuple=True)

    if earliest is None or latest is None:
        print("No data in buffer. Exiting.")
        return

    dtearliest = datetime.datetime.fromtimestamp(earliest, tz=datetime.timezone.utc)
    dtlatest = datetime.datetime.fromtimestamp(latest, tz=datetime.timezone.utc)
    print(f"Buffer holds data from {dtearliest} to {dtlatest}")

    # Truncate to the start of the prior UTC day
    dtstart = dtearliest.replace(hour=0, minute=0, second=0, microsecond=0)

    # Loop over all days in the buffer range, check if the archive file exists and save
    # it if it doesn't
    while dtstart + datetime.timedelta(days=1) < dtlatest:
        dtend = dtstart + datetime.timedelta(days=1)

        filename = f"rfi_{dtstart.strftime('%Y-%m-%d')}.h5"

        path = Path(conf["directory"]) / filename

        if not path.exists():
            print(f"Archiving data from {dtstart} to {dtend} into {path}.")
            archive(path, dtstart=dtstart, dtend=dtend, types=spectrum_types)

        dtstart = dtend
