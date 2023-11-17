"""Table definitions for the Sqlite data buffer."""
from enum import Enum
from typing import TypeVar

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


def close() -> None:
    """Close the database connection."""
    database.close()
