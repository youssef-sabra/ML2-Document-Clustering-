import os
from nlp_clustering import visualization


def test_ensure_dir_filepath(tmp_path):
    out_file = tmp_path / "figs" / "plot.png"
    # Should not raise
    visualization._ensure_dir(str(out_file))
    assert (tmp_path / "figs").exists()


def test_ensure_dir_dirpath(tmp_path):
    out_dir = tmp_path / "figures"
    visualization._ensure_dir(str(out_dir))
    assert out_dir.exists()
