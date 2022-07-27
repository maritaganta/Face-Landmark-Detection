from dataclasses import dataclass
from typing import Dict, List, Optional

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

@st.experimental_singleton(suppress_st_warning=True)
def get_data() -> List[Dict]:
    api_key = st.secrets["api_key"]
    url = f"https://developer.nps.gov/api/v1/parks?api_key={api_key}&limit=500"
    resp = requests.get(url)
    data = resp.json()["data"]
    parks = [park for park in data if park["designation"] == "National Park"]

    for park in parks:
        park["_point"] = Point.from_dict(park)

    return parks

@dataclass
class Point:
    lat: float
    lon: float

    @classmethod
    def from_dict(cls, data: Dict) -> "Point":
        if "lat" in data:
            return cls(float(data["lat"]), float(data["lng"]))
        elif "latitude" in data:
            return cls(float(data["latitude"]), float(data["longitude"]))
        else:
            raise NotImplementedError(data.keys())

    def is_close_to(self, other: "Point") -> bool:
        close_lat = self.lat - 0.0001 <= other.lat <= self.lat + 0.0001
        close_lon = self.lon - 0.0001 <= other.lon <= self.lon + 0.0001
        return close_lat and close_lon


@dataclass
class Bounds:
    south_west: Point
    north_east: Point

    def contains_point(self, point: Point) -> bool:
        in_lon = self.south_west.lon <= point.lon <= self.north_east.lon
        in_lat = self.south_west.lat <= point.lat <= self.north_east.lat

        return in_lon and in_lat

    @classmethod
    def from_dict(cls, data: Dict) -> "Bounds":
        return cls(
            Point.from_dict(data["_southWest"]), Point.from_dict(data["_northEast"])
        )