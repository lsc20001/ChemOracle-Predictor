import os
import math
import tempfile
import pandas as pd
from mendeleev import element
from morfeus import BuriedVolume, Dispersion, Sterimol, read_xyz, XTB

COVALENT_RADII = {"Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.50, "Fe": 1.42, "Co": 1.38, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "H": 0.31, "C": 0.73, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39}
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}

METAL_PROPS = {
    "Co": {"D_P": 55.0, "ar_r": 233.0, "EA_Mt": 15.24},
    "Cr": {"D_P": 82.0, "ar_r": 233.0, "EA_Mt": 15.36},
    "Fe": {"D_P": 62.0, "ar_r": 237.0, "EA_Mt": 3.76},
    "Mn": {"D_P": 68.0, "ar_r": 242.0, "EA_Mt": 0.0},
    "Ni": {"D_P": 49.0, "ar_r": 299.0, "EA_Mt": 26.66}
}

def get_l_trans(atoms, target_metal):
    if len(atoms) < 3: return 0
    metal, h_atom = atoms[0], atoms[1]
    LA = math.sqrt((metal['x'] - h_atom['x'])**2 + (metal['y'] - h_atom['y'])**2 + (metal['z'] - h_atom['z'])**2)
    R1 = COVALENT_RADII.get(metal['ele'], 0)
    for i in range(2, len(atoms)):
        target = atoms[i]
        ele = target['ele']
        if ele not in COVALENT_RADII: continue
        R2 = COVALENT_RADII[ele]
        LB = math.sqrt((metal['x'] - target['x'])**2 + (metal['y'] - target['y'])**2 + (metal['z'] - target['z'])**2)
        LC = math.sqrt((h_atom['x'] - target['x'])**2 + (h_atom['y'] - target['y'])**2 + (h_atom['z'] - target['z'])**2)
        VAR = LA**2 + LB**2 + 1.970 * LA * LB
        R3 = R1 * 1.35 + R2 * 1.35
        if (LB < R3) and (LB < 2.8) and (VAR < LC**2):
            return ATOMIC_NUMBERS.get(ele, 0)
    return 0

def extract_all_16_features(file_content, target_metal, user_ox):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.xyz', delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        lines = file_content.strip().split('\n')
        atoms = [l for l in lines[2:] if l.strip()]
        parsed_atoms = [{'ele': p.split()[0], 'x': float(p.split()[1]), 'y': float(p.split()[2]), 'z': float(p.split()[3])} for p in atoms if p.strip()]

        elements, coordinates = read_xyz(tmp_path)
        
        xtb = XTB(elements, coordinates)
        
        raw_bo = xtb.get_bond_order(1, 2)
        bo1_2 = round(raw_bo - 0.0114, 4) if target_metal.upper() == "MN" else round(raw_bo, 4)
        
        charges = xtb.get_charges()
        charges1 = round(charges[1], 4)
        homo = xtb.get_homo()
        lumo = xtb.get_lumo()
        homo_lumo_gap = round(lumo - homo, 4)
        ip = round(xtb.get_ip(), 4)
        debye = round(math.sqrt(sum(c**2 for c in xtb.get_dipole())), 4)

        bv = BuriedVolume(elements, coordinates, 1, excluded_atoms=[2, 3, 4, 5, 6, 7])
        disp = Dispersion(elements, coordinates)
        sterimol = Sterimol(elements, coordinates, 1, 2)
        
        nca_c_count, nca_n_count = 0, 0
        r1 = COVALENT_RADII.get(target_metal, 0) 
        metal_coord = parsed_atoms[0]
        for target in parsed_atoms[1:]:
            ele2 = target['ele']
            r2 = COVALENT_RADII.get(ele2, 0)
            if r2 == 0: continue
            dist = math.sqrt((metal_coord['x'] - target['x'])**2 + (metal_coord['y'] - target['y'])**2 + (metal_coord['z'] - target['z'])**2)
            r3 = r1 * 1.35 + r2 * 1.35
            if (dist < r3) and (dist < 2.8):
                if ele2 == 'C': nca_c_count += 1
                elif ele2 == 'N': nca_n_count += 1

        props = METAL_PROPS.get(target_metal, {"D_P": 0.0, "ar_r": 0.0, "EA_Mt": 0.0})

        features = {
            'ox': user_ox,  
            'Debye': debye, 
            'bo1_2': bo1_2, 
            'HOMO-LUMO': homo_lumo_gap,
            'ip': ip, 
            'charges1': charges1, 
            'B_1': round(sterimol.B_1_value, 4),
            'B_5': round(sterimol.B_5_value, 4), 
            'P_int2': round(disp.atom_p_int[1], 4),
            'D_P': props["D_P"], 
            'ar_r': props["ar_r"], 
            'BV': round(bv.fraction_buried_volume, 6),
            'EA_Mt': props["EA_Mt"], 
            'NCA_N': nca_n_count, 
            'LT': get_l_trans(parsed_atoms, target_metal),
            'NCA_C': nca_c_count
        }
        return features
    finally:
        os.remove(tmp_path)