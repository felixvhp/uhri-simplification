# UHRI Project

## Repository Structure

```
uhri-dev/
├── code/                # Source code and implementation
│   ├── scripts/            # helper scripts
│   ├── src/                # source code
│   ├── tests/              # tests
├── data/                # Datasets and data resources
├── docs/                # Documentation
│   ├── prototypes/         # Protoypes for external presentation
│   ├── spec/               # Write-up of requirements, specifications, etc.
└── README.md            
```

## Running Tests

```bash
cd code
python -m pytest tests/
# or run a single file directly:
python tests/dataloader_test.py
python tests/test-evaluations.py
```

## Documentation Guidelines

### Code Files

- Include a header comment with file purpose and key dependencies
- Document public functions/classes with descriptions, parameters, and return types
- delete/change comments when updating referenced code
- for each sub-system, keep a "open issues" list / "next work item" (i.e. the place where the next person should resume working)

**Example:**

```python
"""
Pkg #1: Retrieval Implementation [last updated xxx]

Usage: XXX

Open Issues:
- ...

Resumption:
- ...

"""

def extract_filter(
    manual_tag: UHRITag,
    query: str,
) -> UHRIFilter:
    """
    Extract context tags and information from a user query.
    
    Parses the input query string and combines it with manual tags
    to identify relevant UHRI tags and constraints.
    
    Args:
        manual_tag: UHRITag from hard-coded extraction
        query: User query string to parse for tags
        
    Returns:
        UHRIFilter object representing extracted tags and information
    """
```

### Commit Messages

- e.g. `[pkg1] added missing acronyms to vectorization`
