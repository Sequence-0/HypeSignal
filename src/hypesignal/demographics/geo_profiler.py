"""Geographic entity extraction and normalization profiler.

Resolves raw location strings, GPS coordinates, and text mentions into
normalized GeoLocationProfile records with city, state, country, and coordinates.
Combines exact gazetteer lookups, regex pattern parsing, and spaCy NER.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language

from hypesignal.demographics.schemas import GeoLocationProfile
from hypesignal.models.canonical import CanonicalUser, GeoCoordinates

logger = logging.getLogger(__name__)

# US States mapping: 2-letter code to full name
US_STATES: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

US_STATES_LOWER: Dict[str, Tuple[str, str]] = {
    name.lower(): (name, code) for code, name in US_STATES.items()
}
for code, name in US_STATES.items():
    US_STATES_LOWER[code.lower()] = (name, code)

CANADIAN_PROVINCES: Dict[str, str] = {
    "AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba",
    "NB": "New Brunswick", "NL": "Newfoundland and Labrador", "NS": "Nova Scotia",
    "NT": "Northwest Territories", "NU": "Nunavut", "ON": "Ontario",
    "PE": "Prince Edward Island", "QC": "Quebec", "SK": "Saskatchewan",
    "YT": "Yukon",
}
CANADIAN_PROVINCES_LOWER: Dict[str, Tuple[str, str]] = {
    name.lower(): (name, code) for code, name in CANADIAN_PROVINCES.items()
}
for code, name in CANADIAN_PROVINCES.items():
    CANADIAN_PROVINCES_LOWER[code.lower()] = (name, code)

COUNTRIES: Dict[str, Tuple[str, str]] = {
    "us": ("United States", "US"),
    "usa": ("United States", "US"),
    "united states": ("United States", "US"),
    "uk": ("United Kingdom", "GB"),
    "gb": ("United Kingdom", "GB"),
    "united kingdom": ("United Kingdom", "GB"),
    "england": ("United Kingdom", "GB"),
    "scotland": ("United Kingdom", "GB"),
    "wales": ("United Kingdom", "GB"),
    "ca": ("Canada", "CA"),
    "canada": ("Canada", "CA"),
    "au": ("Australia", "AU"),
    "australia": ("Australia", "AU"),
    "de": ("Germany", "DE"),
    "germany": ("Germany", "DE"),
    "fr": ("France", "FR"),
    "france": ("France", "FR"),
    "jp": ("Japan", "JP"),
    "japan": ("Japan", "JP"),
    "in": ("India", "IN"),
    "india": ("India", "IN"),
    "br": ("Brazil", "BR"),
    "brazil": ("Brazil", "BR"),
    "it": ("Italy", "IT"),
    "italy": ("Italy", "IT"),
    "es": ("Spain", "ES"),
    "spain": ("Spain", "ES"),
    "nl": ("Netherlands", "NL"),
    "netherlands": ("Netherlands", "NL"),
    "mx": ("Mexico", "MX"),
    "mexico": ("Mexico", "MX"),
    "sg": ("Singapore", "SG"),
    "singapore": ("Singapore", "SG"),
}

# Major cities gazetteer: key -> (Canonical City, State/Province, Country, CountryCode, (lat, lon))
MAJOR_CITIES: Dict[str, Tuple[str, Optional[str], str, str, Tuple[float, float]]] = {
    "san francisco": ("San Francisco", "California", "United States", "US", (37.7749, -122.4194)),
    "sf": ("San Francisco", "California", "United States", "US", (37.7749, -122.4194)),
    "new york": ("New York", "New York", "United States", "US", (40.7128, -74.0060)),
    "nyc": ("New York", "New York", "United States", "US", (40.7128, -74.0060)),
    "brooklyn": ("New York", "New York", "United States", "US", (40.6782, -73.9442)),
    "manhattan": ("New York", "New York", "United States", "US", (40.7831, -73.9712)),
    "los angeles": ("Los Angeles", "California", "United States", "US", (34.0522, -118.2437)),
    "la": ("Los Angeles", "California", "United States", "US", (34.0522, -118.2437)),
    "chicago": ("Chicago", "Illinois", "United States", "US", (41.8781, -87.6298)),
    "houston": ("Houston", "Texas", "United States", "US", (29.7604, -95.3698)),
    "austin": ("Austin", "Texas", "United States", "US", (30.2672, -97.7431)),
    "dallas": ("Dallas", "Texas", "United States", "US", (32.7767, -96.7970)),
    "buffalo": ("Buffalo", "New York", "United States", "US", (42.8864, -78.8784)),
    "bloomington": ("Bloomington", "Indiana", "United States", "US", (39.1653, -86.5264)),
    "las vegas": ("Las Vegas", "Nevada", "United States", "US", (36.1699, -115.1398)),
    "vegas": ("Las Vegas", "Nevada", "United States", "US", (36.1699, -115.1398)),
    "miami": ("Miami", "Florida", "United States", "US", (25.7617, -80.1918)),
    "atlanta": ("Atlanta", "Georgia", "United States", "US", (33.7490, -84.3880)),
    "new orleans": ("New Orleans", "Louisiana", "United States", "US", (29.9511, -90.0715)),
    "houma": ("Houma", "Louisiana", "United States", "US", (29.5958, -90.7195)),
    "seattle": ("Seattle", "Washington", "United States", "US", (47.6062, -122.3321)),
    "boston": ("Boston", "Massachusetts", "United States", "US", (42.3601, -71.0589)),
    "denver": ("Denver", "Colorado", "United States", "US", (39.7392, -104.9903)),
    "philadelphia": ("Philadelphia", "Pennsylvania", "United States", "US", (39.9526, -75.1652)),
    "philly": ("Philadelphia", "Pennsylvania", "United States", "US", (39.9526, -75.1652)),
    "phoenix": ("Phoenix", "Arizona", "United States", "US", (33.4484, -112.0740)),
    "san diego": ("San Diego", "California", "United States", "US", (32.7157, -117.1611)),
    "san jose": ("San Jose", "California", "United States", "US", (37.3382, -121.8863)),
    "portland": ("Portland", "Oregon", "United States", "US", (45.5152, -122.6784)),
    "detroit": ("Detroit", "Michigan", "United States", "US", (42.3314, -83.0458)),
    "minneapolis": ("Minneapolis", "Minnesota", "United States", "US", (44.9778, -93.2650)),
    "kent": ("Kent", "Washington", "United States", "US", (47.3809, -122.2348)),
    "washington": ("Washington", "District of Columbia", "United States", "US", (38.9072, -77.0369)),
    "washington dc": ("Washington", "District of Columbia", "United States", "US", (38.9072, -77.0369)),
    "dc": ("Washington", "District of Columbia", "United States", "US", (38.9072, -77.0369)),
    "orlando": ("Orlando", "Florida", "United States", "US", (28.5383, -81.3792)),
    "tampa": ("Tampa", "Florida", "United States", "US", (27.9506, -82.4572)),
    "nashville": ("Nashville", "Tennessee", "United States", "US", (36.1627, -86.7816)),
    "charlotte": ("Charlotte", "North Carolina", "United States", "US", (35.2271, -80.8431)),
    "pittsburgh": ("Pittsburgh", "Pennsylvania", "United States", "US", (40.4406, -79.9959)),
    "columbus": ("Columbus", "Ohio", "United States", "US", (39.9612, -82.9988)),
    "indianapolis": ("Indianapolis", "Indiana", "United States", "US", (39.7684, -86.1581)),
    "kansas city": ("Kansas City", "Missouri", "United States", "US", (39.0997, -94.5786)),
    "st. louis": ("St. Louis", "Missouri", "United States", "US", (38.6270, -90.1994)),
    "saint louis": ("St. Louis", "Missouri", "United States", "US", (38.6270, -90.1994)),
    "baltimore": ("Baltimore", "Maryland", "United States", "US", (39.2904, -76.6122)),
    "salt lake city": ("Salt Lake City", "Utah", "United States", "US", (40.7608, -111.8910)),
    # International
    "toronto": ("Toronto", "Ontario", "Canada", "CA", (43.6532, -79.3832)),
    "vancouver": ("Vancouver", "British Columbia", "Canada", "CA", (49.2827, -123.1207)),
    "montreal": ("Montreal", "Quebec", "Canada", "CA", (45.5017, -73.5673)),
    "london": ("London", None, "United Kingdom", "GB", (51.5074, -0.1278)),
    "paris": ("Paris", None, "France", "FR", (48.8566, 2.3522)),
    "berlin": ("Berlin", None, "Germany", "DE", (52.5200, 13.4050)),
    "tokyo": ("Tokyo", None, "Japan", "JP", (35.6762, 139.6503)),
    "sydney": ("Sydney", "New South Wales", "Australia", "AU", (-33.8688, 151.2093)),
    "melbourne": ("Melbourne", "Victoria", "Australia", "AU", (-37.8136, 144.9631)),
    "singapore": ("Singapore", None, "Singapore", "SG", (1.3521, 103.8198)),
    "mumbai": ("Mumbai", "Maharashtra", "India", "IN", (19.0760, 72.8777)),
    "delhi": ("Delhi", "Delhi", "India", "IN", (28.6139, 77.2090)),
    "new delhi": ("Delhi", "Delhi", "India", "IN", (28.6139, 77.2090)),
    "bengaluru": ("Bengaluru", "Karnataka", "India", "IN", (12.9716, 77.5946)),
    "bangalore": ("Bengaluru", "Karnataka", "India", "IN", (12.9716, 77.5946)),
    "dubai": ("Dubai", None, "United Arab Emirates", "AE", (25.2048, 55.2708)),
}


class GeoProfiler:
    """Extracts and normalizes geographic locations from profiles, coordinates, and texts."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize GeoProfiler with loaded spaCy model.
        
        Args:
            spacy_model: Name of spaCy model package to load.
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            logger.warning("Failed to load spaCy model '%s': %s. Initializing blank 'en'.", spacy_model, e)
            self.nlp = spacy.blank("en")

    def _find_nearest_city(
        self, lat: float, lon: float, max_dist_deg: float = 0.8
    ) -> Optional[Tuple[str, Optional[str], str, str, Tuple[float, float]]]:
        """Find the closest major city within max distance in degrees."""
        closest = None
        min_dist = float("inf")
        for city_info in MAJOR_CITIES.values():
            clat, clon = city_info[4]
            # Simple euclidean distance in degrees (sufficient for regional proximity)
            dist = math.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
            if dist < min_dist and dist <= max_dist_deg:
                min_dist = dist
                closest = city_info
        return closest

    def profile_location_string(
        self,
        raw: Optional[str],
        explicit_coords: Optional[GeoCoordinates] = None,
    ) -> GeoLocationProfile:
        """Resolve a raw location string or coordinates into a GeoLocationProfile.
        
        Args:
            raw: Freeform text string like 'San Francisco, CA' or 'UT: 37.77,-122.41'.
            explicit_coords: Explicitly known GeoCoordinates if available.
            
        Returns:
            Normalized GeoLocationProfile.
        """
        raw_clean = (raw or "").strip()

        # Step 1: Explicit coordinates or parseable GPS string
        coords = explicit_coords or (GeoCoordinates.from_raw_string(raw_clean) if raw_clean else None)
        if coords is not None:
            nearest = self._find_nearest_city(coords.latitude, coords.longitude)
            if nearest:
                return GeoLocationProfile(
                    raw_location=raw_clean or f"{coords.latitude},{coords.longitude}",
                    city=nearest[0],
                    state_province=nearest[1],
                    country=nearest[2],
                    country_code=nearest[3],
                    coordinates=coords,
                    confidence=1.0,
                    source="explicit_gps",
                )
            return GeoLocationProfile(
                raw_location=raw_clean or f"{coords.latitude},{coords.longitude}",
                coordinates=coords,
                confidence=1.0,
                source="explicit_gps",
            )

        if not raw_clean:
            return GeoLocationProfile(raw_location=None, confidence=0.0, source="unknown")

        lower = raw_clean.lower()

        # Step 2: Direct match in major cities gazetteer
        if lower in MAJOR_CITIES:
            city, state, country, code, (lat, lon) = MAJOR_CITIES[lower]
            return GeoLocationProfile(
                raw_location=raw_clean,
                city=city,
                state_province=state,
                country=country,
                country_code=code,
                coordinates=GeoCoordinates(latitude=lat, longitude=lon),
                confidence=0.90,
                source="pattern_matched",
            )

        # Step 3: Delimited pattern matching (e.g. 'Austin, TX', 'Miami, Florida', 'Toronto, Canada', 'San Francisco, CA, USA')
        if any(sep in raw_clean for sep in [",", "/", "-", "|"]):
            parts = [p.strip() for p in re.split(r"[,/\-|]+", raw_clean) if p.strip()]
            if len(parts) >= 3:
                city_part = parts[0].lower()
                mid_part = parts[1].lower()
                suffix_part = parts[-1].lower()

                # Check if suffix is a Country (e.g. "USA", "Canada", "UK")
                if suffix_part in COUNTRIES:
                    c_name, c_code = COUNTRIES[suffix_part]
                    state_name = None
                    if mid_part in US_STATES_LOWER:
                        state_name, _ = US_STATES_LOWER[mid_part]
                    elif mid_part in CANADIAN_PROVINCES_LOWER:
                        state_name, _ = CANADIAN_PROVINCES_LOWER[mid_part]

                    city_name = parts[0].strip().title()
                    coords_val = None
                    if city_part in MAJOR_CITIES:
                        m_city, m_state, m_country, m_code, (lat, lon) = MAJOR_CITIES[city_part]
                        if m_code == c_code and (state_name is None or m_state == state_name):
                            city_name = m_city
                            state_name = state_name or m_state
                            coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=state_name,
                        country=c_name,
                        country_code=c_code,
                        coordinates=coords_val,
                        confidence=0.98,
                        source="pattern_matched",
                    )

            if len(parts) >= 2:
                city_part = parts[0].lower()
                suffix_part = parts[-1].lower()

                # 3a. Disambiguation using city context if city is a known major metropolitan area
                if city_part in MAJOR_CITIES:
                    m_city, m_state, m_country, m_code, (lat, lon) = MAJOR_CITIES[city_part]
                    coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    # Does suffix match city's known country or country code? (e.g. "Toronto, CA", "New Delhi, IN", "Berlin, DE", "London, UK")
                    if (
                        suffix_part == m_code.lower()
                        or suffix_part == m_country.lower()
                        or (suffix_part in COUNTRIES and COUNTRIES[suffix_part][1].lower() == m_code.lower())
                    ):
                        return GeoLocationProfile(
                            raw_location=raw_clean,
                            city=m_city,
                            state_province=m_state,
                            country=m_country,
                            country_code=m_code,
                            coordinates=coords_val,
                            confidence=0.95,
                            source="pattern_matched",
                        )

                    # Does suffix match city's known state/province or state code? (e.g. "San Francisco, CA", "Bloomington, IN", "Toronto, ON")
                    m_state_code = None
                    if m_state:
                        if m_state.lower() in US_STATES_LOWER:
                            m_state_code = US_STATES_LOWER[m_state.lower()][1].lower()
                        elif m_state.lower() in CANADIAN_PROVINCES_LOWER:
                            m_state_code = CANADIAN_PROVINCES_LOWER[m_state.lower()][1].lower()

                    if (
                        (m_state and suffix_part == m_state.lower())
                        or (m_state_code and suffix_part == m_state_code)
                    ):
                        return GeoLocationProfile(
                            raw_location=raw_clean,
                            city=m_city,
                            state_province=m_state,
                            country=m_country,
                            country_code=m_code,
                            coordinates=coords_val,
                            confidence=0.95,
                            source="pattern_matched",
                        )

                # 3b. Suffix is a full country name (e.g. "Canada", "India", "France", "Japan")
                if len(suffix_part) > 2 and suffix_part in COUNTRIES:
                    c_name, c_code = COUNTRIES[suffix_part]
                    city_name = city_part.title()
                    coords_val = None
                    state_prov = None
                    if city_part in MAJOR_CITIES and MAJOR_CITIES[city_part][3] == c_code:
                        city_name, state_prov, _, _, (lat, lon) = MAJOR_CITIES[city_part]
                        coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=state_prov,
                        country=c_name,
                        country_code=c_code,
                        coordinates=coords_val,
                        confidence=0.95,
                        source="pattern_matched",
                    )

                # 3c. Suffix is a Canadian province (e.g. "ON", "Ontario", "BC", "Quebec")
                if suffix_part in CANADIAN_PROVINCES_LOWER and suffix_part not in US_STATES_LOWER:
                    prov_name, prov_code = CANADIAN_PROVINCES_LOWER[suffix_part]
                    city_name = city_part.title()
                    coords_val = None
                    if city_part in MAJOR_CITIES and MAJOR_CITIES[city_part][3] == "CA" and MAJOR_CITIES[city_part][1] == prov_name:
                        city_name, _, _, _, (lat, lon) = MAJOR_CITIES[city_part]
                        coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=prov_name,
                        country="Canada",
                        country_code="CA",
                        coordinates=coords_val,
                        confidence=0.95,
                        source="pattern_matched",
                    )

                # 3d. Suffix is an unambiguous country code (e.g. "UK", "GB", "JP", "FR", "AU")
                if suffix_part in COUNTRIES and suffix_part not in US_STATES_LOWER:
                    c_name, c_code = COUNTRIES[suffix_part]
                    city_name = city_part.title()
                    coords_val = None
                    state_prov = None
                    if city_part in MAJOR_CITIES and MAJOR_CITIES[city_part][3] == c_code:
                        city_name, state_prov, _, _, (lat, lon) = MAJOR_CITIES[city_part]
                        coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=state_prov,
                        country=c_name,
                        country_code=c_code,
                        coordinates=coords_val,
                        confidence=0.90,
                        source="pattern_matched",
                    )

                # 3e. Suffix is a US State (e.g. "TX", "Texas", "CA", "California", "IN", "Indiana")
                if suffix_part in US_STATES_LOWER:
                    state_name, state_code = US_STATES_LOWER[suffix_part]
                    city_name = city_part.title()
                    coords_val = None
                    if city_part in MAJOR_CITIES and MAJOR_CITIES[city_part][3] == "US" and MAJOR_CITIES[city_part][1] == state_name:
                        city_name, _, _, _, (lat, lon) = MAJOR_CITIES[city_part]
                        coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=state_name,
                        country="United States",
                        country_code="US",
                        coordinates=coords_val,
                        confidence=0.95,
                        source="pattern_matched",
                    )

                # 3f. Suffix is Canadian Province fallback
                if suffix_part in CANADIAN_PROVINCES_LOWER:
                    prov_name, prov_code = CANADIAN_PROVINCES_LOWER[suffix_part]
                    city_name = city_part.title()
                    coords_val = None
                    if city_part in MAJOR_CITIES and MAJOR_CITIES[city_part][3] == "CA" and MAJOR_CITIES[city_part][1] == prov_name:
                        city_name, _, _, _, (lat, lon) = MAJOR_CITIES[city_part]
                        coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=prov_name,
                        country="Canada",
                        country_code="CA",
                        coordinates=coords_val,
                        confidence=0.95,
                        source="pattern_matched",
                    )

                # 3g. Suffix is remaining Country fallback
                if suffix_part in COUNTRIES:
                    c_name, c_code = COUNTRIES[suffix_part]
                    city_name = city_part.title()
                    coords_val = None
                    state_prov = None
                    if city_part in MAJOR_CITIES and MAJOR_CITIES[city_part][3] == c_code:
                        city_name, state_prov, _, _, (lat, lon) = MAJOR_CITIES[city_part]
                        coords_val = GeoCoordinates(latitude=lat, longitude=lon)

                    return GeoLocationProfile(
                        raw_location=raw_clean,
                        city=city_name,
                        state_province=state_prov,
                        country=c_name,
                        country_code=c_code,
                        coordinates=coords_val,
                        confidence=0.90,
                        source="pattern_matched",
                    )

        # Step 4: Check if raw string is a US State directly (e.g. "California", "Ontario")
        if lower in US_STATES_LOWER:
            state_name, _ = US_STATES_LOWER[lower]
            return GeoLocationProfile(
                raw_location=raw_clean,
                state_province=state_name,
                country="United States",
                country_code="US",
                confidence=0.85,
                source="pattern_matched",
            )
        if lower in CANADIAN_PROVINCES_LOWER:
            prov_name, _ = CANADIAN_PROVINCES_LOWER[lower]
            return GeoLocationProfile(
                raw_location=raw_clean,
                state_province=prov_name,
                country="Canada",
                country_code="CA",
                confidence=0.85,
                source="pattern_matched",
            )
        if lower in COUNTRIES:
            country_name, country_code = COUNTRIES[lower]
            return GeoLocationProfile(
                raw_location=raw_clean,
                country=country_name,
                country_code=country_code,
                confidence=0.85,
                source="pattern_matched",
            )

        # Step 5: spaCy Named Entity Recognition (NER) for GPE / LOC
        doc = self.nlp(raw_clean)
        gpe_entities = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]

        for ent_text in gpe_entities:
            ent_lower = ent_text.lower()
            if ent_lower in MAJOR_CITIES:
                city, state, country, code, (lat, lon) = MAJOR_CITIES[ent_lower]
                return GeoLocationProfile(
                    raw_location=raw_clean,
                    city=city,
                    state_province=state,
                    country=country,
                    country_code=code,
                    coordinates=GeoCoordinates(latitude=lat, longitude=lon),
                    confidence=0.75,
                    source="ner_extracted",
                )
            if ent_lower in US_STATES_LOWER:
                state_name, _ = US_STATES_LOWER[ent_lower]
                return GeoLocationProfile(
                    raw_location=raw_clean,
                    state_province=state_name,
                    country="United States",
                    country_code="US",
                    confidence=0.75,
                    source="ner_extracted",
                )
            if ent_lower in COUNTRIES:
                country_name, country_code = COUNTRIES[ent_lower]
                return GeoLocationProfile(
                    raw_location=raw_clean,
                    country=country_name,
                    country_code=country_code,
                    confidence=0.75,
                    source="ner_extracted",
                )

        if gpe_entities:
            return GeoLocationProfile(
                raw_location=raw_clean,
                city=gpe_entities[0].title(),
                confidence=0.50,
                source="ner_extracted",
            )

        # Fallback
        return GeoLocationProfile(
            raw_location=raw_clean,
            confidence=0.0,
            source="unknown",
        )

    def extract_locations_from_text(self, text: str) -> List[GeoLocationProfile]:
        """Extract and resolve geographical entities mentioned in arbitrary text or tweet.
        
        Args:
            text: Arbitrary post text (e.g. 'Just arrived in San Francisco, next stop London').
            
        Returns:
            List of resolved GeoLocationProfile instances for all detected locations.
        """
        if not text:
            return []

        doc = self.nlp(text)
        profiles: List[GeoLocationProfile] = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                clean = ent.text.strip()
                if clean.lower() not in seen:
                    seen.add(clean.lower())
                    profile = self.profile_location_string(clean)
                    if profile.confidence > 0.0:
                        profiles.append(profile)

        return profiles

    def profile_user(self, user: CanonicalUser) -> GeoLocationProfile:
        """Extract normalized location profile for a CanonicalUser."""
        return self.profile_location_string(
            raw=user.location_raw,
            explicit_coords=user.location_coords,
        )
