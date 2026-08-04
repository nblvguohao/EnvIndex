"""envindex — Paper 2: learned stage-aware environmental index and GxE
predictability boundary across crops.

This package holds the shared code for the EnvIndex project
(protocol_freeze_paper2.md).  Data-acquisition scripts live in scripts/;
environment-id utilities live here.
"""

from envindex.envid import (  # noqa: F401
    KNOWN_SOURCES,
    ParsedEnvironmentId,
    is_environment_id,
    make_environment_id,
    parse_environment_id,
    sanitize_native,
)

__version__ = "0.1.0"
