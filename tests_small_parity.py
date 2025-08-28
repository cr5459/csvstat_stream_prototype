import subprocess, sys, re, os, tempfile

def test_stream_prints_columns_and_counts():
    csv = "a,b\n1,2\n,3\n4,\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(csv); path = f.name
    try:
        out = subprocess.check_output([sys.executable, "csvstat_stream.py", "--force-stream", path], text=True)
        # Expect mode line
        assert "Mode: STREAM" in out
        # Expect numbered column headers in block format
        assert re.search(r'1\. "a"', out)
        assert re.search(r'2\. "b"', out)
        # Expect Row count footer
        assert re.search(r'Row count:\s*3', out)
    finally:
        os.remove(path)
