import pandas as pd


def remove_cmp_formatting(s):
    """
    Used in read_cmp() to remove formatting in .cmp file

    Parameters
    ----------
    s: str
        one line in the file

    Returns
    -------
    list of strings
    """
    for r in (("\t", " "), ("\n", ""), ("elec", "")):
        s = s.replace(*r)
    return s.split()


def read_cmp(file_path):
    """
    Read in Blackrock Microsystems .cmp file into Python

    Parameters
    ----------
    file_path: str
        .cmp file path + name

    Returns
    -------
    df_array: dataframe of shape (num electrodes, 5)
        [col (int), row (int), channel number (str), within_channel_num (int), global electrode number (int)]
    """
    # Open file, remove comments and remove other formatting we don't need
    with open(file_path) as f:
        temp = [line for line in f if not line.startswith("//")]
    clean_lsts = [remove_cmp_formatting(l) for l in temp[1:]]
    df = pd.DataFrame(
        clean_lsts,
        columns=[
            "array_col",
            "array_row",
            "channel_num",
            "within_channel_num",
            "global_enum",
        ],
    ).dropna()

    # Convert columns to integers - errors='igore' return the column unchanged if it cannot be converted to a numeric type
    df_array = df.apply(pd.to_numeric, errors="ignore")

    return df_array
