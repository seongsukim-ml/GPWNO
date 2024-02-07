from mp_api.client import MPRester
from emmet.core.summary import HasProps
import json
from tqdm import tqdm

my_api_key = "pdDir9sfmf4JPMKDSJzF8W7ZJ7CyqYHE"

print("Downloading MP dataset...")

with MPRester(my_api_key) as mpr:
    docs = mpr.materials.summary.search(
        has_props = [HasProps.charge_density], fields=["material_id"],chunk_size=100
    )
    mpids = [doc.material_id for doc in docs]

print("Downloaded {} MP IDs.".format(len(mpids)))

print("Saving MP IDs to JSON file...")

json_str = json.dumps(mpids)

# Save the JSON string to a file
with open('mpids.json', 'w') as f:
    f.write(json_str)
    
    
print("Downloading charge density files...")
path = "/home/holywater2/crystal/dataset_mp"
print("Saving charge density files to {}...".format(path))

with MPRester(my_api_key) as mpr:
    for mp_id in tqdm(mpids):
        chgcar = mpr.get_charge_density_from_material_id(mp_id)
        chgcar.write_file(path + "/" + mp_id + ".chgcar")