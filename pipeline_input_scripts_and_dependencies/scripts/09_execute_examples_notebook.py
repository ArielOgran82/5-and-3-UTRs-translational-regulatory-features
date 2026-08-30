import nbformat as nbf
from nbclient import NotebookClient

REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
NB_PATH = f"{REPO}/examples/predict_new_transcript.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=300, kernel_name="rf_rescue",
                         resources={"metadata": {"path": f"{REPO}/examples"}})
client.execute()
nbf.write(nb, NB_PATH)
print("Executed and saved.")
