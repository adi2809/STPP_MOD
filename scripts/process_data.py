import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gdch.data import process_raw_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Process cleaned_data.csv into GDCH format")
    parser.add_argument("--input", default="dataframes/cleaned_data.csv")
    parser.add_argument("--output-events", default="data/events.csv")
    parser.add_argument("--output-metadata", default="data/metadata.json")
    parser.add_argument("--output-opo-metadata", default="data/opo_metadata.csv")
    parser.add_argument("--output-distance", default="data/distance.npy")
    parser.add_argument("--time-col", default="event_datetime_local")
    parser.add_argument("--opo-col", default="OPO_ENTIRE_NAME_CLEAN")
    parser.add_argument("--lat-col", default="OPO_LAT_ZIP")
    parser.add_argument("--lon-col", default="OPO_LON_ZIP")
    parser.add_argument("--min-time-delta", type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_events), exist_ok=True)
    process_raw_csv(
        input_path=args.input,
        output_events_path=args.output_events,
        output_metadata_path=args.output_metadata,
        output_opo_metadata_path=args.output_opo_metadata,
        output_distance_path=args.output_distance,
        time_col=args.time_col,
        opo_col=args.opo_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        min_time_delta=args.min_time_delta,
    )


if __name__ == "__main__":
    main()
