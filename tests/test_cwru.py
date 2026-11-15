from collections import Counter

from bearing_fault_detection.cwru import CATALOG


def test_catalog_has_40_files():
    assert len(CATALOG) == 40


def test_catalog_class_counts():
    # 4 normal recordings; 3 fault sizes x 4 loads = 12 per fault type
    assert Counter(label for label, _, _ in CATALOG.values()) == {0: 4, 1: 12, 2: 12, 3: 12}


def test_known_file_numbers():
    assert CATALOG[97] == (0, 0.0, 0)      # Normal, 0 hp
    assert CATALOG[105] == (1, 0.007, 0)   # Inner race 0.007", 0 hp
    assert CATALOG[237] == (3, 0.021, 3)   # Outer race 0.021", 3 hp