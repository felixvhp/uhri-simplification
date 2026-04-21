"""Quick tests for the data loader."""
 
from src.Dataloader import load_records
 
DATA_PATH = "../data/search-9_3_2026_592.xlsx"
 
 
def test_loads_all_records():
    records = load_records(DATA_PATH)
    assert len(records) == 43, f"Expected 43 records, got {len(records)}"
 
 
def test_fields_not_empty():
    records = load_records(DATA_PATH)
    for r in records:
        assert r.annotation_id, "Missing annotation_id"
        assert r.text, "Missing text"
        assert r.country, "Missing country"
        assert r.recommending_body, "Missing recommending_body"
        assert r.type, "Missing type"
 
 
def test_prefix_stripped():
    """The '- ' prefix from the export should be cleaned off."""
    records = load_records(DATA_PATH)
    for r in records:
        assert not r.country.startswith("- "), f"Country still has prefix: {r.country}"
        assert not r.recommending_body.startswith("- "), f"Body still has prefix: {r.recommending_body}"
        assert not r.type.startswith("- "), f"Type still has prefix: {r.type}"
 
 
def test_valid_types():
    records = load_records(DATA_PATH)
    valid = {"Recommendations", "Concerns/Observations"}
    for r in records:
        assert r.type in valid, f"Unexpected type: {r.type}"
 
 
def test_properties():
    records = load_records(DATA_PATH)
    recs = [r for r in records if r.is_recommendation]
    obs = [r for r in records if r.is_observation]
    assert len(recs) + len(obs) == len(records), "Every record should be one or the other"
 
 
if __name__ == "__main__":
    tests = [test_loads_all_records, test_fields_not_empty, test_prefix_stripped, test_valid_types, test_properties]
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")