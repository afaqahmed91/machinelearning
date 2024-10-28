import numpy as np
import pandas as pd


def distance(df: pd.DataFrame, lat: float, lon: float) -> float:
    p = 0.017453292519943295
    lat_diff = (df["latitude"] - lat) * p
    lon_diff = (df["longitude"] - lon) * p
    hav = (
        0.5
        - np.cos(lat_diff) / 2
        + np.cos(lat * p) * np.cos(df["latitude"] * p) * (1 - np.cos(lon_diff)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(hav))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by performing various data cleaning and transformation steps.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing water pump data.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with imputed and transformed features.

    Steps:
    - Imputes missing or zero longitude values based on the mean longitude of the region code.
    - Imputes missing or zero gps_height values based on the mean gps_height of the basin.
    - Calculates the center_distance from a fixed point (-6.5, 37.5).
    - Imputes missing or zero population values based on the mean population of the basin.
    - Calculates the age of the water pump based on the date recorded and construction year.
    - Imputes negative or NaN age values with the mean age.
    - Replaces NaN values in 'funder' and 'installer' columns with "Unknown".
    - Replaces NaN values in 'public_meeting' and 'permit' columns with False.
    - Drops unnecessary columns: 'region_code', 'date_recorded', 'construction_year',
      'latitude', 'longitude', 'water_quality', 'extraction_type', 'payment',
      'scheme_management', 'waterpoint_type_group', 'scheme_name', 'amount_tsh'.
    """
    return (
        df.assign(
            # imputed longitude based on region code
            longitude=np.where(
                df["longitude"] == 0,
                df["region_code"].map(
                    df.groupby("region_code")["longitude"].mean().to_dict()
                ),
                df["longitude"],
            ),
            # imputed longitude based on basin
            gps_height=np.where(
                df["gps_height"] == 0,
                df["basin"].map(df.groupby("basin")["gps_height"].mean().to_dict()),
                df["gps_height"],
            ),
            center_distance=lambda x: distance(x, -6.5, 37.5),
            population=np.where(
                df["population"] == 0,
                df["basin"].map(df.groupby("basin")["population"].mean().to_dict()),
                df["population"],
            ),
            age=(
                df["date_recorded"].str.split("-").str[0].astype(int)
                - df["construction_year"].replace(0, np.nan)
            ),
            quantity=df.quantity.replace({"unknown": "dry"}).replace(
                {"dry": 1, "insufficient": 2, "seasonal": 3, "enough": 4}
            ),
            public_meeting=df.public_meeting.fillna(False).astype(int),
            permit=df.permit.fillna(False).astype(int),
            funder=df.funder.fillna("Unknown").pipe(topn, n=10),
            installer=df.installer.fillna("Unknown").replace({"0":"Unknown"}).pipe(topn, n=10),
        )
        .assign(
            age=lambda _df: np.where(
                (_df["age"] < 0) | (_df["age"].isna()), _df["age"].mean(), _df["age"]
            )
        )
        .drop(
            columns=[
                "region_code",
                "date_recorded",
                "construction_year",
                "latitude",
                "longitude",
                "water_quality",
                "extraction_type",
                "payment",
                "scheme_management",
                "waterpoint_type_group",
                "scheme_name",
                "amount_tsh",
                "subvillage",
            ]
        )
    )


def topn(ser: pd.Series, n: int = 5, default: str = "other"):
    """
    Replace values in a pandas Series with 'default' if they are not among the top 'n' most frequent values.

    Parameters:
    ser (pandas.Series): The input pandas Series.
    n (int, optional): The number of top most frequent values to keep. Defaults to 5.
    default (str, optional): The value to replace non-top 'n' values with. Defaults to "other".

    Returns:
    pandas.Series: A pandas Series with non-top 'n' values replaced by 'default'.
    """
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)
