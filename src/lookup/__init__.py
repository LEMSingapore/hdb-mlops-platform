"""Postal code lookup module for resolving Singapore postal codes to addresses."""

from lookup.abbreviations import expand_street_abbreviations
from lookup.postal import AddressInfo, lookup_postal

__all__ = ["AddressInfo", "expand_street_abbreviations", "lookup_postal"]
