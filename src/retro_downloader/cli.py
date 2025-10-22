"""
This command-line interface will retrieve National Water Model version 3.0
Retrospective output and build a WRES-friendly archive of compressed CSVs.

Example usage: retro-downloader -d /path/to/my_archive
"""
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import StrEnum
import getpass
import json
import yaml

import click
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

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

def build_routelink(destination: Path) -> pd.DataFrame:
    """
    Build crosswalk from National Water Model channel feature identifiers to
    USGS site codes.

    Parameters
    ----------
    destination: pathlib.Path
        Destination directory to build local archive of NWM Retrospective
        output in NetCDF and WRES-compatible CSV format.
    
    Returns
    -------
    Resulting crosswalk in a pandas.DataFrame
    """
    # Logger
    logger = get_logger()

    # Check for existing
    ofile = destination / "routelink.parquet"
    if ofile.exists():
        logger.info("Found %s", ofile)
        return pd.read_parquet(ofile, engine="pyarrow")

    # Handle download location
    logger.info("Setting up download directory")
    destination.mkdir(exist_ok=True, parents=True)

    # Process each source
    logger.info("Building routelink")
    dfs = []
    for source in SOURCES:
        # Open dataset
        logger.info("Extracting gages %s", source.url)
        ds = xr.open_dataset(
            source.url,
            backend_kwargs={"storage_options": {"anon": True}},
            engine="zarr"
        )
        gages = ds["gage_id"].to_dataframe().iloc[:, :-1]

        # Close dataset
        ds.close()

        gages["gage_id"] = gages["gage_id"].str.decode("utf-8").str.strip()
        gages = gages.loc[gages["gage_id"] != ""]
        gages["domain"] = source.label
        dfs.append(gages)

    # Merge
    logger.info("Merging routelinks")
    routelink = pd.concat(dfs)

    # Save
    logger.info("Saving %s", ofile)
    routelink.to_parquet(ofile, engine="pyarrow")
    return routelink

def download_to_netcdf(
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

def generate_symlinks(
        destination: Path,
        usgs_rfc_wfo_file: Path
) -> None:
    """
    Generate symlinks to compressed CSVs organized by RFC and WFO.

    Parameters
    ----------
    destination: pathlib.Path
        Path to root download directly.
    usgs_rfc_wfo_file: pathlib.Path
        GeoJSON file containing mapping from USGS site codes to RFCs and WFOs.
    """
    # Logger
    logger = get_logger()

    # Load routelink
    routelink = build_routelink(destination).reset_index().set_index("gage_id")

    # Load mapping
    logger.info("Loading %s", usgs_rfc_wfo_file)
    geomap = gpd.read_file(usgs_rfc_wfo_file)
    geomap = geomap[geomap["STAID"].isin(routelink.index)]
    geomap["feature_id"] = geomap["STAID"].map(routelink["feature_id"])
    geomap["domain"] = geomap["STAID"].map(routelink["domain"])

    # RFCs
    logger.info("Generating RFC symlinks")
    file_log = {
        "metadata": {
            "created_by": getpass.getuser(),
            "created_on": str(pd.Timestamp.utcnow()),
            "model": "National Water Model",
            "model_version": "v3.0",
            "grouped_by": "RFC"
            }
    }
    feature_template = {}
    for (rfc, domain), gdf in geomap.groupby(["RFC_NAME", "domain"]):
        idir = destination / f"csv/{domain}"
        odir = destination / f"rfc/{rfc}"
        odir.mkdir(exist_ok=True, parents=True)
        site_list = file_log.get(rfc, [])
        feature_list = feature_template.get(rfc, [])
        for row in gdf.itertuples():
            target = idir / f"{row.feature_id}_nwm_3_0_retro_wres.csv.gz"
            if not target.exists():
                continue
            link = odir / target.name
            site_list.append(str(link.absolute()))
            feature_list.append({"observed": row.STAID, "predicted": row.feature_id})
            if not link.is_symlink():
                logger.info("%s -> %s", link, target.absolute())
                link.symlink_to(target.absolute())
        file_log[rfc] = site_list
        feature_template[rfc] = feature_list

    # Generate RFC listing
    log_file = destination / "rfc_available_retrospective_csv_files_listing.json"
    logger.info("Writing %s", log_file)
    output = json.dumps(file_log, indent=2)
    with log_file.open("w") as fo:
        fo.write(output)
    feature_mapping = destination / "rfc_feature_mapping.yaml"
    logger.info("Writing %s", feature_mapping)
    with feature_mapping.open("w") as fo:
        yaml.dump(feature_template, fo)

    # WFOs
    logger.info("Generating WFO symlinks")
    file_log = {
        "metadata": {
            "created_by": getpass.getuser(),
            "created_on": str(pd.Timestamp.utcnow()),
            "model": "National Water Model",
            "model_version": "v3.0",
            "grouped_by": "WFO"
            }
    }
    feature_template = {}
    for (wfo, domain), gdf in geomap.groupby(["WFO", "domain"]):
        idir = destination / f"csv/{domain}"
        odir = destination / f"wfo/{wfo}"
        odir.mkdir(exist_ok=True, parents=True)
        site_list = file_log.get(wfo, [])
        feature_list = feature_template.get(wfo, [])
        for row in gdf.itertuples():
            target = idir / f"{row.feature_id}_nwm_3_0_retro_wres.csv.gz"
            if not target.exists():
                continue
            link = odir / target.name
            site_list.append(str(link.absolute()))
            feature_list.append({"observed": row.STAID, "predicted": row.feature_id})
            if not link.is_symlink():
                logger.info("%s -> %s", link, target.absolute())
                link.symlink_to(target.absolute())
        file_log[wfo] = site_list
        feature_template[wfo] = feature_list

    # Generate WFO listing
    log_file = destination / "wfo_available_retrospective_csv_files_listing.json"
    logger.info("Writing %s", log_file)
    output = json.dumps(file_log, indent=2)
    with log_file.open("w") as fo:
        fo.write(output)
    feature_mapping = destination / "wfo_feature_mapping.yaml"
    logger.info("Writing %s", feature_mapping)
    with feature_mapping.open("w") as fo:
        yaml.dump(feature_template, fo)

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
    # download_to_netcdf(destination, ChannelRouteVariable(variable))
    generate_symlinks(destination, Path("usgs_rfc_wfo_mapping.geojson"))

if __name__ == "__main__":
    cli()
