"""Command-line interface."""
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import StrEnum

import click
import xarray as xr
import numpy as np
import pandas as pd

@dataclass
class DataSource:
    """
    Dataclass that stores zarr store details.
    
    Attributes
    ----------
    label: str
        Convenient label used for destination directories.
    url: str
        A s3fs URL string pointing at zarr source.
    """
    label: str
    url: str

SOURCES: list[DataSource] = [
    DataSource("Hawaii", "s3://noaa-nwm-retrospective-3-0-pds/Hawaii/zarr/chrtout.zarr"),
    DataSource("PR", "s3://noaa-nwm-retrospective-3-0-pds/PR/zarr/chrtout.zarr"),
    DataSource("Alaska", "s3://noaa-nwm-retrospective-3-0-pds/Alaska/zarr/chrtout.zarr"),
    DataSource("CONUS", "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"),
    ]
"""List of source labels and zarr URLs."""

def get_logger(name: str = "retro_downloader") -> logging.Logger:
    """
    Generate and return a logger.

    Paramters
    ---------
    name: str, optional, default 'retro_downloader'
        Name of the logger.
    
    Returns
    -------
    Logger
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class ChannelRouteVariable(StrEnum):
    """National Water Model Retrospective xarray.Dataset variables."""
    Q_BTM_VERT_RUNOFF = "qBtmVertRunoff"
    Q_BUCKET = "qBucket"
    Q_SFC_LAT_RUNOFF = "qSfcLatRunoff"
    Q_LATERAL = "q_lateral"
    STREAMFLOW = "streamflow"
    VELOCITY = "velocity"

def main(
        destination: Path,
        variable: ChannelRouteVariable = ChannelRouteVariable.STREAMFLOW
) -> None:
    """
    Download and process National Water Model version 3.0 Retrospective
    output. This specifically retrieves output from the "channel route"
    zarr stores for Alasak, Hawaii, Puerto Rico, and CONUS.

    Parameters
    ----------
    destination: pathlib.Path
        Destination directory to build local archive of NWM Retrospective
        output in NetCDF and WRES-compatible CSV format.
    variable: ChannelRouteVariable, default 'streamflow'
        Variable to retrieve from chrtout.zarr stores.
    """
    # Logger
    logger = get_logger()

    # Process each source
    for source in SOURCES:
        # Open dataset
        logger.info("Opening %s", source.url)
        ds = xr.open_dataset(
            source.url,
            backend_kwargs={"storage_options": {"anon": True}},
            engine="zarr",
            chunks="auto"
        ).unify_chunks()

        # Prepare chunk indexes
        logger.info("Inspecting coordinates")
        chunk_sizes = ds.chunksizes
        time_chunks = np.array_split(ds.time.values, len(chunk_sizes["time"]))
        feature_chunks = np.array_split(ds.feature_id.values, len(chunk_sizes["feature_id"]))
        gage_id_chunks = np.array_split(ds.gage_id.values, len(chunk_sizes["feature_id"]))

        # Handle download location
        logger.info("Setting up download directory")
        raw_dir = destination / f"raw/{source.label}"
        raw_dir.mkdir(exist_ok=True, parents=True)
        csv_dir = destination / f"csv/{source.label}"
        csv_dir.mkdir(exist_ok=True, parents=True)

        # Download and process chunks
        logger.info("Processing chunks")
        logger.info("%d feature chunks", len(feature_chunks))
        blank_gage_code = ''.rjust(15).encode()
        chunk = 0
        for gage_indexes, feature_indexes in zip(gage_id_chunks, feature_chunks):
            # Retrieve chunks with gaged locations
            if np.any(gage_indexes != blank_gage_code):
                # Files to process
                ifiles = []

                # Collect all times for gage set
                logger.info("Retrieving %d time chunks", len(time_chunks))
                for time_indexes in time_chunks:
                    # Check for existing file
                    ofile = raw_dir / f"chunk_{chunk}.nc"
                    chunk += 1
                    if ofile.exists():
                        logger.info("Found %s", ofile)

                        # Add for processing
                        ifiles.append(ofile)
                        continue
                    logger.info("Downloading %s", ofile)

                    # Retrieve individual chunk
                    ds[variable].sel(
                        time=time_indexes,
                        feature_id=feature_indexes[gage_indexes != blank_gage_code]
                    ).chunk({"time": 1, "feature_id": 1}).to_netcdf(ofile)

                    # Add for processing
                    ifiles.append(ofile)
                    break

                # Process
                logger.info("Processing chunks")
                data = pd.concat([xr.open_dataarray(fi).to_dataframe() for fi in ifiles])
                data["gage_id"] = data["gage_id"].str.decode("utf-8").str.strip()
                data = data[data["gage_id"].str.isdigit()].reset_index()
                data["streamflow"] = data["streamflow"].div(0.3048 ** 3.0)
                data["measurement_unit"] = "CFS"
                data["variable_name"] = "streamflow"
                data["timescale_in_minutes"] = 1
                data["timescale_function"] = "UNKNOWN"
                data = data.rename(columns={
                    "time": "value_date",
                    "streamflow": "value",
                    "feature_id": "location"
                })
                data = data[[
                    "value_date",
                    "variable_name",
                    "location",
                    "measurement_unit",
                    "value",
                    "timescale_in_minutes",
                    "timescale_function"
                ]]

                # Write CSVs
                logger.info("Writing CSVs")
                for location, df in data.groupby("location"):
                    cfile = csv_dir / f"{location}_nwm_3_0_retro_wres.csv.gz"
                    logger.info("%s", cfile)
                    df.to_csv(
                        cfile,
                        float_format="%.3f",
                        index=False,
                        date_format="%Y-%m-%dT%H:%M:%SZ",
                        compression="gzip"
                    )
                break

        # Close dataset
        logger.info("Closing dataset")
        ds.close()
        break

@click.command()
@click.option("-d", "--destination", "destination", nargs=1, required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination directory to build archive.")
@click.option("-v", "--variable", "variable", nargs=1,
    type=click.Choice([v.value for v in ChannelRouteVariable]),
    help="Channel route variable. Defaults to 'streamflow'.", default="streamflow")
def cli(
        destination: Path | None = None,
        variable: ChannelRouteVariable = ChannelRouteVariable.STREAMFLOW
) -> None:
    """
    Download and process National Water Model version 3.0 Retrospective
    output. This specifically retrieves output from the "channel route"
    zarr stores for Alasak, Hawaii, Puerto Rico, and CONUS.
    """
    main(destination, ChannelRouteVariable(variable))

if __name__ == "__main__":
    cli()
