import subprocess, sys, re, os, tempfile, textwrap

def test_stream_prints_columns_and_counts():
    csv = "a,b\n1,2\n,3\n4,\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(csv); path = f.name
    out = subprocess.check_output([sys.executable, "csvstat_stream.py", "--force-stream", path], text=True)
    assert "Mode: STREAM" in out
    assert re.search(r"a\tcount=\d+", out)
    assert re.search(r"b\tcount=\d+", out)
    os.remove(path)
