import argparse
import gzip
from pathlib import Path
from urllib.request import urlopen

import numpy as np

SOURCES = {
    "noaa": {
        "year": 2020,
        "url": "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/2020",
        "stations": [
            "994033-99999",
            "724676-93073",
            "722104-92806",
            "723110-13873",
            "724390-93822",
            "744915-14775",
            "725404-04847",
            "723069-93753",
        ],
        "variables": [
            "air_temperature",
            "dew_point",
            "sea_level_pressure",
            "wind_direction",
            "wind_speed",
        ],
        "columns": [4, 5, 6, 7, 8],
        "divisors": [10, 10, 10, 1, 10],
    },
    "uscrn": {
        "year": 2025,
        "url": "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2025",
        "stations": [
            "CRNH0203-2025-AZ_Elgin_5_S",
            "CRNH0203-2025-CA_Bodega_6_WSW",
            "CRNH0203-2025-CO_Boulder_14_W",
            "CRNH0203-2025-FL_Everglades_City_5_NE",
            "CRNH0203-2025-GA_Brunswick_23_S",
            "CRNH0203-2025-IL_Champaign_9_SW",
            "CRNH0203-2025-MA_Blue_Hill_0_W",
            "CRNH0203-2025-MI_Chatham_1_SE",
        ],
        "variables": ["t_hr_avg", "t_max", "t_min", "p_calc", "rh_hr_avg"],
        "columns": [9, 10, 11, 12, 26],
        "divisors": [1, 1, 1, 1, 1],
    },
}


def read_station(path, dataset, spec):
    start = np.datetime64(f"{spec['year']}-01-01T00", "h")
    stop = np.datetime64(f"{spec['year'] + 1}-01-01T00", "h")
    times = np.arange(start, stop, np.timedelta64(1, "h"))
    values = np.zeros((len(times), len(spec["columns"])), dtype=np.float32)
    masks = np.zeros_like(values)
    opener = gzip.open if dataset == "noaa" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < (12 if dataset == "noaa" else 38):
                continue
            if dataset == "noaa":
                year, month, day, hour = map(int, fields[:4])
            else:
                date, clock = fields[1:3]
                year, month, day, hour = (
                    int(date[:4]),
                    int(date[4:6]),
                    int(date[6:8]),
                    int(clock[:2]),
                )
            time = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}", "h")
            index = int((time - start) / np.timedelta64(1, "h"))
            if not 0 <= index < len(times):
                continue
            for variable, (column, divisor) in enumerate(
                zip(spec["columns"], spec["divisors"])
            ):
                raw = float(fields[column])
                # Keep the missing-value rules used for the archived station subsets.
                missing = raw == -9999 if dataset == "noaa" else raw <= -90
                if not missing and np.isfinite(raw):
                    values[index, variable], masks[index, variable] = raw / divisor, 1
    return values, masks, times


def main():
    parser = argparse.ArgumentParser(
        description="Build the fixed eight-station hourly subset"
    )
    parser.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Fetch missing source files from NOAA"
    )
    args = parser.parse_args()
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("output must be a new .npz file")
    spec = SOURCES[args.dataset]
    values, masks = [], []
    for station in spec["stations"]:
        name = (
            f"{station}-{spec['year']}.gz"
            if args.dataset == "noaa"
            else f"{station}.txt"
        )
        path = args.raw_dir / name
        if not path.exists():
            if not args.download:
                parser.error(
                    f"Missing {path}; download it from {spec['url']}/{name} or use --download"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f"{spec['url']}/{name}", timeout=60) as response:
                payload = response.read()
            with path.open("xb") as stream:
                stream.write(payload)
        station_values, station_masks, times = read_station(path, args.dataset, spec)
        if not station_masks.any():
            raise ValueError(f"No observed measurements parsed from {path}")
        values.append(station_values)
        masks.append(station_masks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            values=np.stack(values),
            masks=np.stack(masks),
            timestamps=times.astype("datetime64[s]").astype(str),
            variables=np.array(spec["variables"]),
            station_ids=np.array(spec["stations"]),
        )
    print(f"Saved {args.output}: {len(values)} stations, {len(times)} hours")


if __name__ == "__main__":
    main()
