from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Int64

from data_sources import tennis_source

day = Entity(name="day", join_keys=["day_id"])

tennis_features = FeatureView(
    name="tennis_features",
    entities=[day],
    ttl=timedelta(days=365),
    schema=[
        Field(name="outlook", dtype=Int64),
        Field(name="temperature", dtype=Int64),
        Field(name="humidity", dtype=Int64),
        Field(name="wind", dtype=Int64),
        Field(name="play", dtype=Int64),
    ],
    source=tennis_source,
)
