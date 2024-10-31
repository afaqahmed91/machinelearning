import numpy as np
import pandas as pd


def distance(df: pd.DataFrame, lat: float, lon: float) -> float:
    """
    Calculate the distance between a given latitude and longitude and each point in a DataFrame.

    This function uses the Haversine formula to compute the great-circle distance between two points
    on the Earth's surface specified by latitude and longitude.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
    lat (float): Latitude of the reference point.
    lon (float): Longitude of the reference point.

    Returns:
    float: The distance in kilometers between the reference point and each point in the DataFrame.
    """
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
    Preprocess the input DataFrame by imputing missing values, creating new features,
    and dropping unnecessary columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing raw data.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with imputed values, new features,
                  and selected columns.

    Imputations and Transformations:
    - longitude: Impute missing values (0) based on the mean longitude of the region code.
    - gps_height: Impute missing values (0) based on the mean gps_height of the basin.
    - center_distance: Calculate the distance from a central point (-6.5, 37.5).
    - population: Impute missing values (0 or 1) based on the mean population of the basin.
    - age: Calculate the age of the water point based on the recorded date and construction year.
    - quantity: Replace 'unknown' with 'dry' and convert to categorical type.
    - public_meeting: Fill missing values with False and convert to integer type.
    - permit: Fill missing values with False and convert to integer type.
    - funder: Fill missing values with 'Unknown', keep top 10 funders, and convert to categorical type.
    - installer: Fill missing values with 'Unknown', replace '0' with 'Unknown', keep top 10 installers, and convert to categorical type.
    - ground_water: Replace 'groundwater' with 1, 'surface' with 0, 'unknown' with 1, and convert to integer type.
    - age: Recalculate age, replacing negative or NaN values with the mean age.

    Dropped Columns:
    - region_code, date_recorded, construction_year, water_quality, extraction_type,
      payment, scheme_management, waterpoint_type_group, scheme_name, amount_tsh,
      subvillage, num_private, district_code, recorded_by, wpt_name, ward,
      quantity_group, source_class, source_type.
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
                (df["population"] == 0) | (df["population"] == 1),
                df["basin"].map(
                    df.groupby("basin")["population"].mean().astype(int).to_dict()
                ),
                df["population"],
            ),
            age=(
                df["date_recorded"].str.split("-").str[0].astype(int)
                - df["construction_year"].replace(0, np.nan)
            ),
            quantity=df.quantity.replace({"unknown": "dry"})
            # .replace({"dry": 1, "insufficient": 2, "seasonal": 3, "enough": 4})
            .astype("category"),
            public_meeting=df.public_meeting.fillna(False).astype(int),
            permit=df.permit.fillna(False).astype(int),
            funder=df.funder.fillna("Unknown").pipe(topn, n=10).astype("category"),
            installer=df.installer.fillna("Unknown")
            .replace({"0": "Unknown"})
            .pipe(topn, n=10)
            .astype("category"),
            ground_water=df.source_class.replace(
                {"groundwater": 1, "surface": 0, "unknown": 1}
            ).astype(int),
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
                # "latitude",
                # "longitude",
                "water_quality",
                "extraction_type",
                "payment",
                "scheme_management",
                "waterpoint_type_group",
                "scheme_name",
                "amount_tsh",
                "subvillage",
                "num_private",
                "district_code",
                "recorded_by",
                "wpt_name",
                "ward",
                "quantity_group",
                "source_class",
                "source_type",
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


def preprocess_y(df):
    return (
        df["status_group"]
        .replace({"functional": 2, "non functional": 0, "functional needs repair": 1})
        .values
    )
