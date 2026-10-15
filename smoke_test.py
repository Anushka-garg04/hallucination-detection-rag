"""Headless Streamlit smoke test (no browser needed)."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120)
at.run()
assert not at.exception, f"App crashed on load: {at.exception}"

btn = [b for b in at.button if b.label == "Generate and Verify"][0]
btn.click()
at.run()
assert not at.exception, f"App crashed on run: {at.exception}"

metrics = {m.label: m.value for m in at.metric}
print("METRICS:", metrics)
assert any("Verification" in k for k in metrics), "missing verification metric"
assert any("Hallucination" in k for k in metrics), "missing hallucination metric"
print("SMOKE TEST PASSED")
