"""Label space shared by every model in the study.

38 named diseases are organised into six clinically motivated groups. Twenty-one
ultra-rare classes (five or fewer training images each) are pooled into a single
``G_Rare`` target, giving 39 scored targets in total.
"""

GROUPS = {
    "G1_Vascular":     ["DR", "HR", "HTN", "BRVO", "CRVO", "TV", "CWS", "PRH", "MCA"],
    "G2_Exudative":    ["ARMD", "DN", "EDN", "CSR", "CME", "RPEC", "MHL", "ERM", "PT"],
    "G3_Disc":         ["ODC", "GLA", "ODP", "ODE", "AION", "TD", "ST"],
    "G4_Structural":   ["MYA", "TSLN", "LS", "RT", "RD", "CF"],
    "G5_Media":        ["MH", "CAT"],
    "G6_Inflammatory": ["CRS", "RS", "MS", "AH", "RP"],
}

# Ultra-rare classes pooled into one target (not sub-classified).
RARE = ["VH", "PLQ", "CSC", "GRT", "OPDM", "RHL", "BRAO", "CNV", "ON", "RTR",
        "MNF", "CRAO", "HPED", "ME", "SOFE", "CB", "PTCR", "CL", "IIH", "VS", "ODPM"]
RARE_COL = "G_Rare"

DISEASES = [d for members in GROUPS.values() for d in members]   # 38 named diseases
TARGETS = DISEASES + [RARE_COL]                                   # 39 scored targets

# Stage-2 router outputs: six clinical groups + the pooled rare group.
GROUP_COLS = list(GROUPS) + [RARE_COL]
# Short keys used for the Stage-3 group classifiers.
GKEY = {f"G{i}": g for i, g in enumerate(GROUPS, start=1)}       # "G1" -> "G1_Vascular"

FULL_NAME = {
    "DR": "Diabetic Retinopathy", "HR": "Hypertensive Retinopathy", "HTN": "Hypertension",
    "BRVO": "Branch Retinal Vein Occlusion", "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels", "CWS": "Cotton Wool Spots", "PRH": "Preretinal Haemorrhage",
    "MCA": "Macroaneurysm", "ARMD": "Age-related Macular Degeneration", "DN": "Drusen",
    "EDN": "Exudates", "CSR": "Central Serous Retinopathy", "CME": "Cystoid Macular Edema",
    "RPEC": "RPE Changes", "MHL": "Macular Hole", "ERM": "Epiretinal Membrane",
    "PT": "Parafoveal Telangiectasia", "ODC": "Optic Disc Cupping", "GLA": "Glaucoma",
    "ODP": "Optic Disc Pallor", "ODE": "Optic Disc Edema",
    "AION": "Anterior Ischemic Optic Neuropathy", "TD": "Tilted Disc", "ST": "Optociliary Shunt",
    "MYA": "Myopia", "TSLN": "Tessellation", "LS": "Laser Scars", "RT": "Retinal Traction",
    "RD": "Retinal Detachment", "CF": "Choroidal Folds", "MH": "Media Haze", "CAT": "Cataract",
    "CRS": "Chorioretinitis", "RS": "Retinitis", "MS": "Macular Scar", "AH": "Asteroid Hyalosis",
    "RP": "Retinitis Pigmentosa", "G_Rare": "Pooled ultra-rare pathology",
}

SOURCES = ("odir", "rfmid1", "rfmid2")


def source_of(name: str) -> str:
    """Infer the source dataset from the namespaced image identifier."""
    name = str(name)
    if name.startswith("rfmid_1"):
        return "rfmid1"
    if name.startswith("rfmid_2"):
        return "rfmid2"
    return "odir"
