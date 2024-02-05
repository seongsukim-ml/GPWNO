import argparse
from mp_api.client import MPRester
from emmet.core.summary import HasProps
import json
from tqdm import tqdm
import os

# Create an argument parser
parser = argparse.ArgumentParser(description="Download MP dataset")

# fmt:off
parser.add_argument('--download_mpids', type=bool, help='Download MP ids', default=False)
parser.add_argument('--start', type=int, help='Start index for downloading MP ids', default=0)
parser.add_argument('--end', type=int, help='End index for downloading MP ids', default=-1)
parser.add_argument('--chgcar_path', type=str, help='Path to save charge density files', default="../../dataset_mp/data")
parser.add_argument('--use_except', type=bool, help='Flag to use excepts log', default=False)
parser.add_argument('--mpids_path', type=str, help='Path to save MP ids JSON file', default="./mpids_query.json")
# fmt:on


# Parse the command line arguments
args = parser.parse_args()

my_api_key = "pdDir9sfmf4JPMKDSJzF8W7ZJ7CyqYHE"

print("[I] Downloading MP dataset...")

mpids_path = args.mpids_path

# Downloading and saving mpids
if args.download_mpids:
    with MPRester(my_api_key) as mpr:
        docs = mpr.materials.summary.search(
            has_props=[HasProps.charge_density], fields=["material_id"], chunk_size=100
        )
        mpids = [doc.material_id for doc in docs]

    print("[I] Downloaded {} MP IDs.".format(len(mpids)))
    print("[I] Saving MP IDs to JSON file...")

    json_str = json.dumps(mpids)

    # Save the JSON string to a file
    with open(mpids_path, "w") as f:
        f.write(json_str)

# Loading mpids
with open(mpids_path, "r") as file:
    mpids = json.load(file)

print("[I] Len mpids:", len(mpids))

# Downloading charge density files
print("[I] Downloading charge density files...")
print("[I] Saving charge density files to {}...".format(args.chgcar_path))

# Create the directory if it doesn't exist
os.makedirs(args.chgcar_path, exist_ok=True)

saved_files = os.listdir(args.chgcar_path)
saved_files_wo_ext = []
for i in saved_files:
    saved_files_wo_ext.append(i.split(".")[0])

print("[I] Len saved_files:", len(saved_files))

cnt = 1

if not os.path.isfile("mp-ids-except.txt"):
    open("mp-ids-except.txt", "w")
f = open("mp-ids-except.txt", "r+")
Lines = f.readlines()

except_files_list = []
for line in Lines:
    except_files_list.append(line.split(" ")[0])

print("[I] Len except_files_list:", len(except_files_list))

f = open("mp-ids-except.txt", "a+")

if args.end == -1:
    args.end = len(mpids)

with MPRester(my_api_key) as mpr:
    for mp_id in tqdm(mpids[args.start : args.end]):
        print("[I]", cnt, "/", args.end - args.start, " & ", mp_id)
        if mp_id in saved_files_wo_ext:
            print("[I]", mp_id, "exists")
            cnt += 1
            continue
        if os.path.isfile(args.chgcar_path + "/" + mp_id + ".chgcar"):
            print("[I]", mp_id, "exists as file")
            cnt += 1
            continue
        if args.use_except and mp_id in except_files_list:
            print("[I]", mp_id, "in excepts log")
            cnt += 1
            continue
        try:
            chgcar = mpr.get_charge_density_from_material_id(mp_id)
        except Exception as e:
            f.write(mp_id + " " + str(e) + "\n")
            print("[Error] " + mp_id + " " + str(e))
        else:
            chgcar.write_file(args.chgcar_path + "/" + mp_id + ".chgcar")
        cnt += 1
