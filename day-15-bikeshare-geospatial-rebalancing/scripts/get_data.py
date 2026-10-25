"""Download public data (Capital Bikeshare trips, NOAA weather, DEM tiles) into data/raw/ and refresh data/sample/.

Falls back to the SYNTHETIC generator automatically if the trip archive is unreachable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bikeshare import data, geo  # noqa: E402

if __name__ == "__main__":
    acq = data.acquire(force_synthetic="--synthetic" in sys.argv)
    stations, prov = geo.add_spatial_features(acq.stations, try_osm=True)
    if not acq.provenance["trips"]["synthetic"]:
        data.write_samples(acq.trips, stations, acq.weather)
    print(json.dumps({**acq.provenance, "spatial": prov}, indent=2))
