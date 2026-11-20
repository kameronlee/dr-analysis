"""Catalog verified source files and render a local review page without outcomes.

This deliberately does not select a primary projection, merge masks, freeze a
calibration set, or create the pipeline's analysis manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from retina_fragility.config import load_config


PROJECTIONS = {
    "oct_full": "OCT(FULL)", "oct_ilm_opl": "OCT(ILM_OPL)",
    "oct_opl_bm": "OCT(OPL_BM)", "octa_full": "OCTA(FULL)",
    "octa_ilm_opl": "OCTA(ILM_OPL)", "octa_opl_bm": "OCTA(OPL_BM)",
}
LABELS = {
    "artery": ("GT_Artery", ".bmp"), "cavf": ("GT_CAVF", ".bmp"),
    "capillary": ("GT_Capillary", ".bmp"), "faz": ("GT_FAZ", ".bmp"),
    "faz3d": ("GT_FAZ3D", ".mat"), "large_vessel": ("GT_LargeVessel", ".bmp"),
    "layers": ("GT_Layers", ".mat"), "vein": ("GT_Vein", ".bmp"),
}


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    manifests = root / "data/manifests"
    raw = Path("data/raw/octa500/extracted")
    verified = {}
    for name in ("Label", "OCTA_3mm_part3", "OCTA_6mm_part8"):
        audit = json.loads((manifests / f"{name}_extraction.json").read_text())
        for record in audit["records"]:
            verified[record["member"]] = record

    rows = []
    for fov in (3, 6):
        cohort = f"OCTA_{fov}mm"
        metadata_member = f"{cohort}/Text labels.xlsx"
        frame = pd.read_excel(root / raw / metadata_member)
        for subject in frame.to_dict("records"):
            sid = str(subject["ID"])
            row = {
                "source_subject_id": sid, "image_id": f"OCTA500_{sid}",
                "dataset": "OCTA500", "fov_mm": fov,
                "pixel_size_um": cfg["data"]["expected_geometry"][str(fov)]["pixel_size_um"],
                "disease": subject["Disease"], "age": subject["Age"],
                "sex": subject["Sex"], "eye": subject["OS/OD"],
                "metadata_path": str(raw / metadata_member),
                "metadata_sha256": verified[metadata_member]["sha256"],
                "primary_projection_selected": False,
            }
            members = {
                key: f"{cohort}/Projection Maps/{group}/{sid}.bmp"
                for key, group in PROJECTIONS.items()
            }
            members.update({key: f"Label/{group}/{sid}{extension}"
                            for key, (group, extension) in LABELS.items()})
            for key, member in members.items():
                record = verified[member]
                path = root / raw / member
                if not path.is_file() or path.stat().st_size != record["bytes"]:
                    raise ValueError(f"Source file missing or size changed: {member}")
                row[key + "_path"] = str(raw / member)
                row[key + "_sha256"] = record["sha256"]
            rows.append(row)
    catalog = pd.DataFrame(rows).sort_values(["fov_mm", "source_subject_id"])
    if len(catalog) != 500 or catalog.image_id.duplicated().any():
        raise ValueError("Unexpected OCTA-500 subject inventory")
    catalog.to_csv(manifests / "octa500_source_catalog.csv", index=False)

    seed = cfg["project"]["base_seed"]
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}|octa500-source-review|{row['image_id']}".encode()).hexdigest())
    page = root / "outputs/octa500_source_review/index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    view, key_rows = [], []
    for n, row in enumerate(ordered, 1):
        review_id = f"OCTA-review-{n:03d}"
        visible = {"review_id": review_id, "fov_mm": row["fov_mm"], "images": []}
        for key in ["octa_full", "octa_ilm_opl", "octa_opl_bm", "capillary",
                    "large_vessel", "faz", "oct_full", "oct_ilm_opl", "oct_opl_bm"]:
            visible["images"].append({"label": key.replace("_", " "),
                                      "src": "../../" + quote(row[key + "_path"], safe="/")})
        view.append(visible)
        key_rows.append({"review_id": review_id, "image_id": row["image_id"],
                         "reviewer": "", "reviewed_at": "", "notes": ""})
    pd.DataFrame(key_rows).to_csv(manifests / "octa500_source_review_key.csv", index=False)
    payload = json.dumps(view).replace("<", "\\u003c")
    page.write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCTA-500 source review</title><style>
body{font:16px system-ui;margin:24px;background:#f3f4f6;color:#182433}
header{position:sticky;top:0;background:#f3f4f6;padding:12px 0;z-index:1}
h1{font-size:24px;margin:0}p{max-width:78ch}button,select{font:inherit;padding:8px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
figure{margin:0;background:white;padding:12px;border:1px solid #ccd3dc}
img{width:100%;height:auto;image-rendering:auto}figcaption{padding:8px 0}
</style><header><h1>OCTA-500 source review</h1>
<p>Original projections and supplied binary masks. Disease labels and demographics
are hidden. These are source previews, not graph quality-control decisions.
All projections are shown; the primary projection has not been selected.</p>
<button id="prev">Previous</button> <select id="subject"></select>
<button id="next">Next</button><p id="status"></p></header><main id="images"></main>
<script type="application/json" id="records">""" + payload + """</script><script>
const rows=JSON.parse(document.getElementById('records').textContent);
const select=document.getElementById('subject');let index=0;
for(const [i,row] of rows.entries()){const option=document.createElement('option');
option.value=i;option.textContent=row.review_id;select.append(option)}
function show(i){index=Math.max(0,Math.min(rows.length-1,i));select.value=index;
const row=rows[index];document.getElementById('status').textContent=
`${row.review_id} · ${row.fov_mm} mm · ${index+1} of ${rows.length}`;
const main=document.getElementById('images');main.replaceChildren();
for(const item of row.images){const fig=document.createElement('figure');
const caption=document.createElement('figcaption');caption.textContent=item.label;
const img=document.createElement('img');img.src=item.src;img.alt=item.label;
fig.append(caption,img);main.append(fig)}
document.getElementById('prev').disabled=index===0;
document.getElementById('next').disabled=index===rows.length-1}
select.onchange=()=>show(Number(select.value));
document.getElementById('prev').onclick=()=>show(index-1);
document.getElementById('next').onclick=()=>show(index+1);show(0);
</script></html>""")
    audit = {
        "n_subjects": len(rows), "n_source_files_per_subject": len(PROJECTIONS) + len(LABELS),
        "base_seed": seed, "review_order": "SHA256(base_seed|octa500-source-review|image_id)",
        "disease_labels_in_review_payload": False, "human_reviews_completed": 0,
        "primary_projection_selected": False, "analysis_manifest_created": False,
        "clinical_features_computed": False,
        "catalog_sha256": hashlib.sha256((manifests / "octa500_source_catalog.csv").read_bytes()).hexdigest(),
        "review_html_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
        "source_hash_provenance": "Verified extraction audits; file presence and size rechecked during catalog creation",
    }
    (page.parent / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Cataloged {len(rows)} subjects and {len(rows)*14} source-file references.")
    print(f"Blinded source preview: {page.relative_to(root)}")


if __name__ == "__main__":
    main()
