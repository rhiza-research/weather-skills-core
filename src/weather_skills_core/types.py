"""Named constants for the ``@weather_skill`` declaration surface.

Envelope types plus the toggle mode strings, so a declaration references a
constant whose definition it can jump to and whose alternatives are
enumerable, rather than repeating a bare string.
"""

#: Spatial dims ``latitude``/``longitude`` (aliases accepted) with a ``time`` dim.
GRIDDED = "gridded"
#: A ``step`` (lead time) dim plus a scalar ``time`` coord for the init date.
FORECAST = "forecast"
#: A ``station_id`` dim with 1-D ``latitude``/``longitude`` coords and a ``time`` dim.
STATION = "station"
#: No spatial coords at all -- what collapsing latitude and longitude leaves.
SERIES = "series"

#: Every envelope type. As an ``input_type`` it accepts all four and still
#: validates the input as whichever one it turns out to be; as an
#: ``output_type`` it is the union of all four.
ALL = (GRIDDED, FORECAST, STATION, SERIES)

#: ``output_type`` of a skill that writes a PNG rather than a Zarr store.
PNG = "png"

#: ``bbox`` modes: the flag is mandatory, or accepted and defaulted to None.
REQUIRED = "required"
OPTIONAL = "optional"

#: ``variable`` modes: one value, or a repeatable flag collecting a list.
SINGLE = "single"
REPEAT = "repeat"
