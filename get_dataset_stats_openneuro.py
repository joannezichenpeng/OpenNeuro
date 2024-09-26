import argparse
import functools
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from tenacity import retry, wait_exponential

# Constants
ENDPOINT = "https://openneuro.org/crn/graphql"
MRI_MODALITY = "MRI"
EEG_LIKE_MODALITIES = ("EEG", "MEG", "iEEG")


initial_query = """
query ($after: String) {
  datasets(after: $after) {
    edges {
      node {
        metadata {
          modalities
          datasetId
          datasetName
          datasetUrl
          openneuroPaperDOI
          species
          seniorAuthor
          studyDesign
          studyDomain
          studyLongitudinal
          tasksCompleted
          trialCount
        }
        latestSnapshot {
          tag
        }
        name
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
  participantCount
}
"""


snapshot_query = """
query snapshotFiles($datasetId: ID!, $tree: String, $tag: String!) {
    snapshot(datasetId: $datasetId, tag: $tag) {
        files(tree: $tree) {
            id
            key
            filename
            size
            directory
            annexed
            urls
        }
    }
}
"""


def custom_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def graphql_query(query, variables=None):
    """
    I had to resort to using a closure because @retry in the root doesn't play well with
    joblib (complains about pickling).
    """

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def _graphql_query(query, variables=None):
        response = requests.post(
            ENDPOINT, json={"query": query, "variables": variables}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed with status code {response.status_code}")

    return _graphql_query(query, variables)


def fetch_relevant_dataset_metadata():
    """
    Fetch all datasets metadata from OpenNeuro which have the right modalities.
    """
    relevant_datasets = []
    has_next_page = True
    cursor = None

    all_modalities = set([MRI_MODALITY]).union(set(EEG_LIKE_MODALITIES))

    while has_next_page:
        print(f"Fetching batch of dataset descriptions with cursor {cursor}")
        variables = {"after": cursor} if cursor else {}
        response = graphql_query(initial_query, variables)
        datasets = response["data"]["datasets"]["edges"]

        # Add datasets with MRI modality to the list
        relevant_datasets += [
            node
            for node in datasets
            if (
                set(node["node"]["metadata"]["modalities"]).intersection(all_modalities)
            )
        ]

        # Update cursor and check for next page
        page_info = response["data"]["datasets"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        cursor = page_info["endCursor"] if has_next_page else None

    return relevant_datasets


def extract_timebase(data):
    """
    Make a best effort to extract the timebase from a tsv file.
    """
    if isinstance(data, dict):
        if "Units" in data:
            return data["Units"]
        else:
            return "s"
    else:
        # Actually a string
        if "milli" in data:
            return "ms"
        else:
            return "s"


def find_root_events(dataset_id, tag):
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tag": tag}
    )

    files = files_response["data"]["snapshot"]["files"]
    if files is None:
        warnings.warn(
            f"No files found for dataset {dataset_id} with tag {tag}"
        )
        return []

    roots_events = [
        file
        for file in files
        if not file["directory"] and file["filename"].endswith("_events.tsv")
    ]

    return roots_events


def find_subjects(dataset_id, tag):
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tag": tag}
    )

    files = files_response["data"]["snapshot"]["files"]
    if files is None:
        return [], None

    # Select all the directories within
    subjects = [
        file
        for file in files
        if file["directory"] and file["filename"].startswith("sub-")
    ]

    # Now find units
    event_file = [
        file
        for file in files
        if not file["directory"] and file["filename"].endswith("events.json")
    ]

    if event_file and len(event_file) == 1:
        # Load the events file to understand the units
        try:
            event_response = requests.get(event_file[0]["urls"][0])
        except requests.exceptions.ConnectionError:
            warnings.warn(
                f"Could not connect to remote file {event_file[0]['filename']}"
            )
            return subjects, None

        try:
            event_content = event_response.content.decode("utf-8")
            event_metadata = json.loads(event_content)
        except UnicodeDecodeError:
            warnings.warn(
                f"Could not decode content of {event_file[0]['filename']} as UTF-8"
            )
            return subjects, None
        except json.JSONDecodeError:
            warnings.warn(
                f"Could not parse JSON content of {event_file[0]['filename']}"
            )
            return subjects, None

        try:
            onset_unit = extract_timebase(event_metadata.get("onset", {}))
            duration_unit = extract_timebase(event_metadata.get("duration", {}))

            multipliers = {
                "s": 1,
                "second": 1,
                "seconds": 1,
                "ms": 1e-3,
                "millisecond": 1e-3,
                "milliseconds": 1e-3,
            }

            return subjects, (
                multipliers.get(onset_unit, 1),
                multipliers.get(duration_unit, 1),
            )
        except (KeyError, AttributeError):
            return subjects, None

    return subjects, None


def find_target_dir(root, dataset_id, tree, tag, target="func"):
    """
    Find a target directory within a root directory.

    Params:
        root: the root directory to search
        dataset_id: the dataset id
        tree: the tree id
        tag: the tag of the snapshot of the dataset
        target: the target directory to search for
    """
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tree": tree, "tag": tag}
    )

    # Check if files is None
    files = files_response["data"]["snapshot"]["files"]
    if files is None:
        warnings.warn(
            f"No files found for dataset {dataset_id} with tree {tree} and tag {tag}"
        )
        return []

    # Proceed safely knowing files is not None
    func_dirs = [
        file
        for file in files
        if file["directory"] and file["filename"] == target
    ]
    if len(func_dirs) == 1:
        return [(f"{root}/{target}", func_dirs[0])]
    else:
        sess_dirs = [
            file
            for file in files
            if file["directory"] and file["filename"].startswith("ses-")
        ]
        if sess_dirs:
            return sum(
                [
                    find_target_dir(
                        root + "/" + sess_dir["filename"],
                        dataset_id,
                        sess_dir["id"],
                        tag,
                        target,
                    )
                    for sess_dir in sess_dirs
                ],
                [],
            )
        else:
            return []


def get_duration_from_tsv(events_files, type="task-fmri", units=None):
    """
    This gets the duration of an experiment by looking at the times between the start
    and end of the events files. It doesn't work for resting state scans. It can
    sometimes break when units other than seconds are used, hence the units argument.
    """
    @functools.lru_cache(maxsize=128)
    def cached_fetch(url):
        """
        Cache the fetch of a url.
        """
        try:
            response = requests.get(url)
            content = response.content.decode("utf-8")
            return content
        except UnicodeDecodeError:
            warnings.warn(f"Could not decode content of {url} as UTF-8")
            return ""
        except (requests.exceptions.ConnectionError, UnicodeDecodeError):
            warnings.warn(f"Could not connect to remote file {url} and decode")
            return ""
        except Exception as e:
            warnings.warn(f"An unexpected error occurred while fetching {url}: {e}")
            return ""

    durations = []
    for events_file in events_files:
        events_response = cached_fetch(events_file["urls"][0])
        if not events_response:
            warnings.warn(f"Empty or invalid events file {events_file['filename']}")
            continue

        try:
            events_df = pd.read_csv(io.StringIO(events_response), sep="\t")
        except pd.errors.ParserError:
            warnings.warn(f"Could not parse {events_file['filename']} as TSV")
            continue
        except Exception as e:
            warnings.warn(f"An error occurred while parsing {events_file['filename']}: {e}")
            continue

        if units is None:
            # Assume unit seconds.
            onset_mult, duration_mult = (1, 1)
        else:
            onset_mult = units[0]
            duration_mult = units[1]

        def safe_cast(arr):
            results = []
            for x in arr:
                try:
                    x = float(x)
                except ValueError:
                    x = np.nan
                results.append(x)
            return results

        try:
            events_df.onset = safe_cast(events_df.onset.tolist())
            events_df.dropna(subset=["onset"], inplace=True)
        except AttributeError:
            warnings.warn(f"Empty or malformed events file {events_file['filename']}")
            continue

        # Sort the events by onset
        events_df = events_df.sort_values("onset")

        try:
            first_event = events_df.iloc[0]
            last_event = events_df.iloc[-1]

            try:
                events_df["duration"] = events_df["duration"].fillna(0)
                last_event_duration = float(last_event["duration"])
            except (KeyError, IndexError, TypeError, ValueError):
                last_event_duration = 0

            if np.isnan(last_event_duration):
                last_event_duration = 0

            # Calculate duration
            durations.append((
                last_event["onset"] - first_event["onset"]
            ) * onset_mult + last_event_duration * duration_mult)
        except (KeyError, IndexError, TypeError) as e:
            warnings.warn(f"Could not compute duration from {events_file['filename']}: {e}")

    total_duration = sum(durations)
    if any([x > 4 * 3600 for x in durations]) and total_duration > 100 * 3600:
        # If one of the tasks is longer than 4 hours;
        # and the total duration is longer than 100 hours,
        # it's safe to assume that the units were actually milliseconds.
        total_duration = total_duration / 1000

    return {"total_duration": total_duration, "type": type}


def estimate_directory_size(root_events_files, dataset_id, tree, tag, units, data_type):
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tree": tree, "tag": tag}
    )

    files = files_response["data"]["snapshot"]["files"]
    if files is None:
        warnings.warn(
            f"No files found for dataset {dataset_id} with tree {tree} and tag {tag}"
        )
        return {"total_duration": 0, "type": f"rs-{data_type.lower()}", "nbytes": 0, "nfiles": 0}

    # Select all the files either finishing with _bold.nii.gz or _events.tsv
    data_files = {}
    events_files = {}
    if files_response["data"]["snapshot"]["files"] is None:
        return {"total_duration": 0, "type": "rs-fmri", "nbytes": 0, "nfiles": 0}

    endings = ["_bold.nii.gz", 
               "_bold.nii", 
               "_bold.nii.gz", 
               "_meg.fif", 
               "_eeg.edf", 
               "_eeg.vhdr", 
               "_eeg.set", 
               "_eeg.fdt", 
               "_eeg.bdf"]

    for file in files_response["data"]["snapshot"]["files"] + root_events_files:
        if any(file["filename"].endswith(x) for x in endings):
            which_ending = [x for x in endings if file["filename"].endswith(x)][0]
            begin = file["filename"][: -len(which_ending)]
            if "_split-" in begin:
                begin = begin[: begin.index("_split-")]
            if "_run-" in begin:
                begin = begin[: begin.index("_run-")]
            if "_echo-" in begin:
                begin = begin[: begin.index("_echo-")]
            data_files[begin] = file

        elif file["filename"].endswith("_events.tsv"):
            begin = file["filename"][: -len("_events.tsv")]
            events_files[begin] = file

    # Check that each data file has a corresponding events file
    nbytes = sum([x["size"] for x in files_response["data"]["snapshot"]["files"]])
    nfiles = len(data_files)

    matches = []
    for data_key, data_values in data_files.items():
        for events_key, events_values in events_files.items():
            # Check if the data and events files have the same prefix or suffix
            if data_key.endswith(events_key) or events_key.startswith(data_key):
                matches.append((data_key, events_key))
                break

    # TODO: Add a codepath for MEG data files in FieldTrip format: 
    # https://www.fieldtriptoolbox.org/getting_started/ctf/
    # As well as one for resting-state fMRI scans.

    if len(matches) == 0:
        # This is likely a resting state scan, do not attempt to estimate duration.
        return {
            "total_duration": 0,
            "nbytes": nbytes,
            "nfiles": nfiles,
            "type": f"rs-{data_type.lower()}",
        }
    else:
        # This is likely a task-based scan, estimate the duration from the tsv.
        normalized_type = f'task-{data_type.lower()}'
        data = get_duration_from_tsv(
            [events_files[x] for _, x in matches], 
            type=normalized_type, 
            units=units)
        return {**data, "nbytes": nbytes, "nfiles": nfiles}


def get_one_dataset(dataset_id, tag, modalities):
    print(f"Processing {dataset_id} {tag}")
    f = open(f"{OUTPUT_DIR}/openneuro_{dataset_id}.jsonl", "a")

    subjects, units = find_subjects(dataset_id, tag)
    # Find functional data for that modality
    root_events_files = find_root_events(
        dataset_id, tag
    )

    for subject in subjects:
        for modality in modalities:
            total_duration = 0
            total_size = 0
            nfiles = 0
            data_type = ""

            modality_map = {"MRI": "func",
                            "EEG": "eeg",
                            "MEG": "meg"}
        
            try:
                modality_dir = modality_map[modality]
            except KeyError:
                modality_dir = "func"

            func_dirs = find_target_dir(
                f"{dataset_id}/{subject['filename']}",
                dataset_id,
                subject["id"],
                tag,
                modality_dir,
            )

            if not func_dirs:
                continue

            for _, func_dir in func_dirs:
                results = estimate_directory_size(
                    root_events_files, dataset_id, func_dir["id"], tag, units, modality
                )
                total_size += results["nbytes"]
                total_duration += results["total_duration"]
                nfiles += results["nfiles"]
                data_type = results["type"]

            json.dump(
                {
                    "dataset_id": dataset_id,
                    "subject": subject["filename"],
                    "total_duration": total_duration,
                    "bytes": total_size,
                    "data_type": data_type,
                    "files": nfiles,
                },
                f,
                default=custom_serializer,
            )
            f.write("\n")
    f.close()


if __name__ == "__main__":
    # Fetch all relevant datasets
    # Get all the relevant datasets and dump them into one file
    description = """Reads OpenNeuro dataset information and dump into jsonl files.
This script reads all the datasets from OpenNeuro which have the right modalities
and dumps the relevant information into jsonl files. The information includes
the dataset id, the subject id, the total duration of the scan, the total size of
the scan in bytes, the type of data (task-based or resting state), and the number
of files in the scan. The script can be run in parallel to speed up the process.

It uses the OpenNeuro GraphQL API to fetch the relevant information. It estimates
the duration of the scan by looking at the events tsv file. This is not a perfect
method and can sometimes break, hence some post-processing is required.

https://docs.openneuro.org/api.html
"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--output_dir', help='Enable parallel processing', default='results')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir

    try:
        os.makedirs(OUTPUT_DIR)
    except FileExistsError:
        pass

    # Use cache to avoid fetching the same data again
    if os.path.exists(f"{OUTPUT_DIR}/openneuro_metadata.json"):
        with open(f"{OUTPUT_DIR}/openneuro_metadata.json", "r") as f:
            relevant_datasets = json.load(f)
        print("Using cached metadata")
    else:
        relevant_datasets = fetch_relevant_dataset_metadata()
        with open(f"{OUTPUT_DIR}/openneuro_metadata.json", "w") as f:
            json.dump(relevant_datasets, f)

    tasks = []
    for dataset in relevant_datasets:
        dataset_id = dataset["node"]["metadata"]["datasetId"]
        tag = dataset["node"]["latestSnapshot"]["tag"]
        modalities = dataset["node"]["metadata"]["modalities"]

        tasks.append(delayed(get_one_dataset)(dataset_id, tag, modalities))

    if args.parallel:
        results = Parallel(n_jobs=-1)(tasks)
    else:
        for func, args, kwargs in tasks:
            func(*args, **kwargs)