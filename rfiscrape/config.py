"""Tools for loading the configuration from files or CLI args."""
import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Param(Generic[T]):
    """Define a parameter we need to infer."""

    def __init__(
        self,
        ptype: T,
        default: T,
        description: str,
        conv: Callable[[Any], T] | None = None,
    ):
        self.ptype = ptype
        self.default = default
        self.description = description
        self.conv = conv

    def get_value(self, value: str) -> T | None:
        """Get the value with any conversions applied."""
        if value is None:
            return None
        if self.conv:
            return self.conv(value)
        return self.ptype(value)


schema = {
    "common": {
        "buffer": Param(str, None, "The path to the ringbuffer file to use."),
    },
    "collector": {
        "port": Param(int, 8464, "The port for the collector to listen on."),
        "window": Param(int, 5, "The length of the assembly window."),
        "time": Param(
            int, 3600 * 24 * 7, "The maximum length of the buffer in seconds.",
        ),
        "purge": Param(
            float,
            0.1,
            (
                "The frequency with which to purge old entries. If less than 1 it is "
                "interpreted as a fraction of the buffer length, if > 1 it is "
                "interpreted as being an interval in seconds."
            ),
        ),
    },
    "server": {
        "port": Param(int, 8465, "The port for the server to listen on."),
    },
    "archiver": {
        "directory": Param(str, None, "The path to archive data to."),
        "types": Param(
            list[str],
            ["STAGE_1", "STAGE_2"],
            (
                "The spectrum types to archive. On the command line use a comma "
                "separated list."
            ),
            lambda x: x.split(",") if isinstance(x, str) else x,
        ),
    },
}


def process_args_and_config(  # noqa: PLR0913
    prog: str,
    description: str,
    schema: dict,
    configbase: str | None = None,
    sections: list[str] | None = None,
    extra_args: list[tuple[str, dict]] | None = None,
) -> dict:
    """Load config and process any command line arguments and return the final config.

    Parameters
    ----------
    prog
        Name of the program for the help message.
    description
        A short description of the program.
    schema
        The configuration schema. A series of nested dictionaries with terminal `Param`
        entries.
    configbase
        Name to use when resolving the config files.
    sections
        Schema sections to map to the root level.
    extra_args
        A set of extra CLI arguments to add. Each list item is the argument name and a
        set of kwargs to feed to `ArgumentParser.add_argument`.

    Returns
    -------
    conf
        The final config.
    """
    # Build a dict of the fuly qualified config parameters to use
    flattened_params = _flatten_dict(_get_sections(schema, sections))

    # Create an argument parser with the overriding arguments
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--config-file",
        type=str,
        help="A configuration file to use which overrides any other locations.",
    )
    for argname, param in flattened_params.items():
        parser.add_argument(f"--{argname}", type=str, help=param.description)

    for argname, kwargs in extra_args:
        parser.add_argument(argname, **kwargs)

    args = parser.parse_args()

    file_config = load_standard_config_files(
        configbase or prog, args.config_file, sections=sections,
    )

    return resolve_config(flattened_params, vars(args), file_config)


def load_standard_config_files(
    name: str, extra_file: str | None = None, sections: list[str] | None = None,
) -> list[dict]:
    """Load config files from the standard locations.

    This will read any files located at the extra_file path, and then
    ~/.config/<NAME>/<NAME>.conf, /etc/xdg/<NAME>/<NAME>.conf, /etc/<NAME>/<NAME>.conf.

    Parameters
    ----------
    name
        The basename to use for the file.
    extra_file
        An arbitrary path that is read and returned first.
    sections
        Sections to extract from the config file. These are extracted and merged into
        the top level.

    Returns
    -------
    list
        A list of the config entries as dictionaries. Missing files are omitted.
    """
    config_files = [
        f"~/.config/{name}/{name}.conf",
        f"/etc/xdg/{name}/{name}.conf",
        f"/etc/{name}/{name}.conf",
    ]
    if extra_file:
        config_files = [extra_file, *config_files]

    config = []

    for cfile in config_files:
        # Expand the configuration file path
        abspath = Path(cfile).expanduser().absolute()

        if not abspath.exists():
            continue

        logger.debug(f"Loading config file {abspath}")

        with abspath.open("r") as fh:
            conf = yaml.safe_load(fh)

        # Extract and merge any sections
        conf = _get_sections(conf, sections)
        config.append(conf)

    return config


def resolve_config(
    params: dict[str, Param],
    cli_args: dict,
    file_config: list[dict],
) -> dict:
    """Resolve all source to a final config."""
    # Filter out unknown params
    allowed_keys = set(params.keys())

    def _filter_dict(d: dict[str, Any]) -> dict[str, any]:
        return {k: v for k, v in d.items() if k in allowed_keys and v is not None}

    # Initialise with the defaults
    resolved_params = {name: p.default for name, p in params.items()}

    # The go through the files (backwards to maintain the overriding precedence)
    for conf in reversed(file_config):
        resolved_params |= _filter_dict(_flatten_dict(conf))

    # Then finally override with any CLI args. Don't filter this one in case any extra
    # arguments are added.
    resolved_params |= cli_args

    # Ensure that any type conversion is done
    return {
        k: (params[k].get_value(v) if k in params else v)
        for k, v in resolved_params.items()
    }


# create_argparser("a", "d", ["common", "archiver"]).parse_args()
# conf = create_argparser("a", "d")
# print(conf)


def _flatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Flatten nested dictionaries into a single layer.

    The final keys are separated by `sep`.
    """
    stack = [("", d)]

    flat_dict = {}

    # Walk through nested dictionary putting items onto a stack for processing
    while stack:
        stem, entry = stack.pop()

        # If a dict, put all it's items onto the stack (backwards to preserve the order)
        if isinstance(entry, dict):
            for key in reversed(entry):
                if not isinstance(key, str):
                    raise TypeError("Only string keys are supported.")
                if sep in key:
                    raise ValueError("String cannot contain the separator.")

                newkey = f"{stem}{sep}{key}" if stem else key

                stack.append((newkey, entry[key]))
        else:
            # Otherwise terminal items are placed in the final dictionary
            flat_dict[stem] = entry

    return flat_dict


def _get_sections(d: dict, sections: list[str] | None) -> dict:
    """Extract the named section keys from d and return the merged result."""
    if sections is None:
        return d

    new_dict = {}

    for section in sections:
        if section in d:
            new_dict |= d[section]

    return new_dict
