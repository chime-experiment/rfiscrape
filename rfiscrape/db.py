"""Table definitions for the Sqlite data buffer."""
from enum import Enum
from typing import TypeVar

import numpy as np
import peewee as pw

# Sqlite database model
database = pw.SqliteDatabase(None)

E = TypeVar("E", bound=Enum)

__schema_version__ = "2023.11"


class EnumField(pw.SmallIntegerField):
    """An Enum like field for Peewee.

    Taken from: https://github.com/coleifer/peewee/issues/630#issuecomment-459404401
    """

    def __init__(self, choices: type[E], *args: tuple, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.choices = choices

    def db_value(self, value: E) -> int:
        """Convert to the DB type."""
        return value.value

    def python_value(self, value: int) -> E:
        """Convert to the Python type."""
        return self.choices(value)


class BaseModel(pw.Model):
    """Base model class."""

    class Meta:
        """Meta info."""

        database = database


class SpectrumType(Enum):
    """An enum type for the different sorts of spectrum."""

    STAGE_1 = 1
    STAGE_2 = 2


class EncodingType(Enum):
    """An enum type for the different ways the data could be encoded."""

    RAW = 0
    LOG_BITSHUFFLE = 1


class RFIData(BaseModel):
    """The file owning user.

    Attributes
    ----------
    id
        Primary key.
    timestamp
        The UTC Unix timestamp as a float64.
    freq_chunk
        The frequency chunk ID. The interpretation of this is spectrum and encoding
        dependent.
    spectrum_type
        The type of spectrum. See the `SpectrumType` enum for the definitions.
    encoding_type
        The type of encoding of the spectrum. This determines how to deserialise the
        data blob.
    data
        The data as a blob.
    """

    id = pw.IntegerField(primary_key=True)  # noqa: A003
    timestamp = pw.FloatField(index=True)

    freq_chunk = pw.SmallIntegerField()

    spectrum_type = EnumField(choices=SpectrumType)
    encoding_type = EnumField(choices=EncodingType)

    data = pw.BlobField()


def connect(filename: str, readonly: bool = True) -> None:
    """Connect to the database.

    Parameters
    ----------
    filename
        The path to the database file.
    readonly
        If set, open the file in read-only mode. If not set, open in read-write mode and
        ensure all the tables are created.
    """
    uri = "file:" + filename

    if readonly:
        uri += "?mode=ro"

    database.init(
        uri,
        pragmas={
            "foreign_keys": 1,
            "journal_mode": "wal",
            "synchronous": "off",
            "temp_store": "memory",
            "mmap_size": 2**28,
            "cache_size": -(2**15),
        },
    )

    # Ensure all the tables are created
    if not readonly:
        database.create_tables(BaseModel.__subclasses__(), safe=True)


def decode(spec: SpectrumType, enc: EncodingType, data: bytes) -> np.ndarray:
    """Decode the data in the RFIData result."""
    if enc == EncodingType.RAW and spec in (SpectrumType.STAGE_1, SpectrumType.STAGE_2):
        return np.frombuffer(data, dtype=np.float32, count=-1)

    raise ValueError("Cannot decode data.")


def fetch_rfi(
    start_time: float,
    end_time: float,
    spec_type: SpectrumType,
    freq_start: float | None = None,
    freq_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read RFI information from the buffer db.

    Parameters
    ----------
    start_time, end_time
        The start and end time to fetch as UTC Unix times.
    spec_type
        Which spectrum to fetch.
    freq_start, freq_end
        The frequency range to fetch in MHz.

    Returns
    -------
    time
        The times in the range.
    freq
        The frequencies in the range.
    data
        The 2D dataset. Missing data is marked with np.nan.
    """
    query = RFIData.select().where(RFIData.spectrum_type == spec_type)
    query = query.where(RFIData.timestamp >= start_time, RFIData.timestamp < end_time)

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

    if chunks:
        freq = np.arange(
            chunks[0] * chunksize, (chunks[-1] + 1) * chunksize, dtype=np.int32
        )
    else:
        freq = np.zeros((0,), dtype=np.int32)

    output_data = np.full(
        (len(timestamps), len(chunks), chunksize), fill_value=np.nan, dtype=np.float32
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


def close() -> None:
    """Close the database connection."""
    database.close()
