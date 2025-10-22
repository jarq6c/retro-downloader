# retro-downloader
Download National Water Model Retrospective Output.

## Installation
```bash
# Retrieve source code
$ git clone https://github.com/jarq6c/retro-downloader.git
$ cd retro-downloader

# Create and activate python environment, requires python >= 3.13
$ python3 -m venv env
$ source env/bin/activate
(env) $ python3 -m pip install --upgrade pip wheel

# Install retro-downloader
(env) $ python3 -m pip install .
```

## Usage
```console
Usage: retro-downloader [OPTIONS]

  Download and process National Water Model version 3.0 Retrospective output.
  This specifically retrieves output from the "channel route" zarr stores for
  Alasak, Hawaii, Puerto Rico, and CONUS.

Options:
  -d, --destination DIRECTORY     Destination directory to build archive.
                                  [required]
  -v, --variable [qBtmVertRunoff|qBucket|qSfcLatRunoff|q_lateral|streamflow|velocity]
                                  Channel route variable. Defaults to
                                  'streamflow'.
  -m, --mapping FILE              GeoJSON mapping from USGS sites codes to RFC
                                  and WFO. Default
                                  usgs_rfc_wfo_mapping.geojson
  --help                          Show this message and exit.
```

# Example
```bash
(env) $ retro-downloader -d /path/to/my_archive
```
