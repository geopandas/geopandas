import sys
import os
import pytest


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)


from _version import render, COVERAGE_FLAGS


def test_render_with_error():
   pieces = {"error": "some error", "long": "abcdef"}
   result = render(pieces, "")
   assert result["version"] == "unknown"
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] is None
   assert result["error"] == "some error"
   assert result["date"] is None


def test_render_default():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None}
   result = render(pieces, "")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_branch():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None}
   result = render(pieces, "pep440-branch")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_post_branch():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
                "date": None}
   result = render(pieces, "pep440-post-branch")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_pre():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None}
   result = render(pieces, "pep440-pre")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_post():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None}
   result = render(pieces, "pep440-post")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_old():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None}
   result = render(pieces, "pep440-old")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_describe():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None}
   result = render(pieces, "git-describe")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_pep440_describe_long():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0", "distance": 0,
               "date": None, "short" : "abc"}
   result = render(pieces, "git-describe-long")
   assert result["version"] is not None
   assert result["full-revisionid"] == "abcdef"
   assert result["dirty"] == False
   assert result["error"] is None


def test_render_unknown_style():
   pieces = {"error": None, "long": "abcdef", "dirty": False, "closest-tag": "v1.0.0"}
   try:
       render(pieces, "unknown")
   except ValueError as e:
       assert str(e) == "unknown style 'unknown'"




def print_percentage(flags):
   total_flags = len(flags)
   active_flags = sum(flags.values())


   coverage_percentage = (active_flags / total_flags) * 100


   print(f"Coverage: {coverage_percentage}%\n")


def print_coverage(flags):
   for branch, hit in flags.items():
       print(f"{branch} was {'hit' if hit else 'not hit'}")


test_render_with_error()
test_render_default()
test_render_pep440_branch()
test_render_pep440_post_branch()
test_render_pep440_pre()
test_render_pep440_post()
test_render_pep440_old()
test_render_pep440_describe()
test_render_pep440_describe_long()
test_render_unknown_style()


print_coverage(COVERAGE_FLAGS)
print_percentage(COVERAGE_FLAGS)
