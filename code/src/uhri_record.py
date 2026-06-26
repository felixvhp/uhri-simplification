"""
Pkg 3: UHRIRecord dataclass 

The common record type feed to the LLM, one instance per row of the
UHRI export, either a recommendation or a concluding observation.

Usage:
    Built by Dataloader.load_records() from the OHCHR xlsx/json export.
    Consumed by pkg3 (simplification)

Open Issues:
    - Only captures 5 of the 16 export columns. Themes, Affected Persons, SDGs
      etc. are dropped. Fine for simplification; not fine if retrieval ever
      needs to filter by theme.
    - No validation and edge case handling
    - More methods could be useful further down the the road

Resumption:
    If you add fields, update Dataloader.load_records too.
    If you're converting any field to an enum, grep for the string literals in
    simplification.py before renaming anything.
"""

from dataclasses import dataclass
 
@dataclass
class UHRIRecord:
    """A single record (recommendation or observation) from the UHRI database."""
 
    annotation_id: str
    text: str
    country: str
    recommending_body: str
    type: str 
 
    @property
    def is_recommendation(self) -> bool:
        return self.type == "Recommendations"
 
    @property
    def is_observation(self) -> bool:
        return self.type == "Concerns/Observations"
 