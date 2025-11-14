import pandas as pd


def file_to_pandas(file):
    """Read a GESLA data file into a pandas.DataFrame object. Metadata is
    returned as a pandas.Series object.
    Args:
        filename (string): name of the GESLA data file. Do not prepend path.
        return_meta (bool, optional): determines if metadata is returned as
            a second function output. Defaults to True.
    Returns:
        pandas.DataFrame: sea-level values and flags with datetime index.
        pandas.Series: record metadata. This return can be excluded by
            setting return_meta=False.
    """


    with open(file, "r") as f:
        data = pd.read_csv(
            f,
            skiprows=41,
            names=["date", "time", "sea_level", "qc_flag", "use_flag"],
            sep="\s+",
            parse_dates=[[0, 1]],
            index_col=0,
        )
    return data
