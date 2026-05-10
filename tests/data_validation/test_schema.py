import pandas as pd
from pandera import Check, Column, DataFrameSchema

schema = DataFrameSchema(
    {
        "day": Column(int, Check.isin(range(1, 15))),
        "outlook": Column(str, Check.isin(["Sunny", "Overcast", "Rain"])),
        "temperature": Column(str, Check.isin(["Hot", "Mild", "Cool"])),
        "humidity": Column(str, Check.isin(["High", "Normal"])),
        "wind": Column(str, Check.isin(["Weak", "Strong"])),
        "play": Column(str, Check.isin(["Yes", "No"])),
    }
)


def test_tennis_csv_schema():
    df = pd.read_csv("data/tennis.csv")
    schema.validate(df)
