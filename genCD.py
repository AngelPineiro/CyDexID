from rdkit import Chem
from rdkit.Chem import AllChem
import re
import os

def gen_smiles(monomer_num):
    """
    Genera el SMILES de un monómero con etiquetas incrementadas según su posición.
    monomer_num: número del monómero en la cadena (1, 2, 3, etc)
    """
    # SMILES base con etiquetas del 1-7
    base_smiles = "[C@@H:1]([O:8])1O[C@H:5]([C:6]O)[C@@H:4]([O:7])[C@H:3](O)[C@H:2]1O"
    
    if monomer_num == 1:
        return base_smiles
        
    # Para monómeros posteriores, incrementar las etiquetas
    offset = (monomer_num - 1) * 8
    new_smiles = base_smiles
    
    # Reemplazar cada etiqueta con su versión incrementada en orden ascendente
    for i in range(1, 9):  # De 1 a 8 en orden ascendente
        new_tag = f":{i + offset}]"
        new_smiles = new_smiles.replace(f":{i}]", new_tag)

    # Reemplazar los números de cierre de anillo de manera única
    # Escapamos los caracteres especiales en la expresión regular
    new_ring_num = str(monomer_num)
    new_smiles = re.sub(r'\]1O\[', f']{new_ring_num}O[', new_smiles)

    return new_smiles

def gen_rxn_idx(pos):
    """
    Genera los números de mapeo para enlazar el monómero anterior con el actual.
    pos: número del monómero actual (comienza en 2)
    """
    if pos < 2:
        raise ValueError("La posición debe ser al menos 2 para generar un enlace.")
    
    c4_map = 7 + (pos - 2) * 8  # 7, 15, 23, ...
    c1_map = 1 + (pos - 1) * 8  # 1, 9, 17, ...
    o1_map = 8 + (pos - 1) * 8  # 8, 16, 24, ...
    
    return c1_map, c4_map, o1_map

def get_atom_indices_by_map_num_and_symbol(mol, map_num_symbol_list):
    """
    Retorna los índices de átomos que tienen los números de mapeo y símbolos especificados.
    mol: objeto RDKit Mol
    map_num_symbol_list: lista de tuplas con (map_num, símbolo), e.g., [(7, 'O'), (8, 'C')]
    """
    indices = []
    for atom in mol.GetAtoms():
        for mn, sym in map_num_symbol_list:
            if atom.GetAtomMapNum() == mn and atom.GetSymbol() == sym:
                indices.append(atom.GetIdx())
                break  # Evita múltiples matches
    return indices

def generar_polimero(num_monomeros):
    """
    Genera un polímero con el número especificado de monómeros.
    num_monomeros: número de unidades monoméricas deseadas
    """
    if num_monomeros < 1:
        raise ValueError("El número de monómeros debe ser al menos 1")
    
    # Generar primer monómero
    mol_actual = Chem.MolFromSmiles(gen_smiles(1))
    Chem.AssignStereochemistry(mol_actual, cleanIt=True, force=True)
    Chem.SanitizeMol(mol_actual)
    print("Monómero 1: ", Chem.MolToSmiles(mol_actual, isomericSmiles=True))
    
    for i in range(2, num_monomeros + 1):
        nuevo_monomero = Chem.MolFromSmiles(gen_smiles(i))
        Chem.AssignStereochemistry(nuevo_monomero, cleanIt=True, force=True)
        Chem.SanitizeMol(nuevo_monomero)
        print(f"\nMonómero {i}: ", Chem.MolToSmiles(nuevo_monomero, isomericSmiles=True))
        
        combined_mol = Chem.CombineMols(mol_actual, nuevo_monomero)
        editable_mol = Chem.EditableMol(combined_mol)
        
        # Obtener los números de mapeo para enlazar
        c1_map, c4_map, o1_map = gen_rxn_idx(i)
        print("\nNúmeros de mapeo para enlace: ", c1_map, c4_map, o1_map)
        
        # Obtener los índices de los átomos a enlazar considerando el símbolo
        # Asumiendo que c1_map corresponde a un átomo de oxígeno y c4_map a un átomo de carbono
        indices = get_atom_indices_by_map_num_and_symbol(combined_mol, [(o1_map, 'O')])
        if len(indices) != 1:
            raise ValueError(f"No se encontró el átomo con mapeo y símbolo 'O'")
        idx3 = indices[0]
        print("Índice de átomo para eliminar: ", idx3)
        
        # Agregar el enlace
        editable_mol.RemoveAtom(idx3)  # Eliminar el átomo idx3
        combined_mol = editable_mol.GetMol()
        indices = get_atom_indices_by_map_num_and_symbol(combined_mol, [(c1_map, 'C'), (c4_map, 'O')])
        idx1, idx2 = indices
        editable_mol.AddBond(idx1, idx2, order=Chem.rdchem.BondType.SINGLE)  # Formar enlace entre idx1 e idx2
        
        # Obtener el nuevo objeto molecular
        mol_actual = editable_mol.GetMol()
        Chem.AssignStereochemistry(mol_actual, cleanIt=True, force=True)
        Chem.SanitizeMol(mol_actual, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
        print(f"\nPolímero {i}: ", Chem.MolToSmiles(mol_actual, isomericSmiles=True))
        
    # Cerrar el polímero
    c4_map= 7 + (num_monomeros - 1) * 8
    c1_map= 1
    o1_map = 8

    indices = get_atom_indices_by_map_num_and_symbol(mol_actual, [(o1_map, 'O')])
    if len(indices) != 1:
        raise ValueError(f"No se encontró el átomo con mapeo y símbolo 'O'")
    idx3 = indices[0]
    print("Índice de átomo para eliminar: ", idx3)
    editable_mol.RemoveAtom(idx3)  # Eliminar el átomo idx3
    mol_actual = editable_mol.GetMol()
    
    indices = get_atom_indices_by_map_num_and_symbol(mol_actual, [(c1_map, 'C'), (c4_map, 'O')])
    if len(indices) != 2:
        raise ValueError(f"No se encontraron los átomos con mapeo {c1_map} y {c4_map} y símbolos 'C' y 'O'")
    idx1, idx2 = indices
    editable_mol.AddBond(idx1, idx2, order=Chem.rdchem.BondType.SINGLE)
    mol_actual = editable_mol.GetMol()
    Chem.AssignStereochemistry(mol_actual, cleanIt=True, force=True)
    Chem.SanitizeMol(mol_actual, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
    print(f"\nPolímero final cerrado de {num_monomeros} unidades: ", Chem.MolToSmiles(mol_actual, isomericSmiles=True))
    
    return mol_actual

def remove_atom_map_numbers(mol):
    """
    Elimina todos los números de mapeo de átomos en una molécula RDKit.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): Objeto molecular de RDKit.

    Returns:
    rdkit.Chem.rdchem.Mol: Molécula sin números de mapeo.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def guardar_estructura(mol, prefijo="polimero"):
    """
    Guarda la estructura en formatos PDB y SMILES, tanto la versión inicial 3D como la optimizada
    """
    mol = remove_atom_map_numbers(mol)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    #mol = Chem.AddHs(mol)
    mol.RemoveAllConformers()
    
    # Generar estructura 3D inicial
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if embed_status != 0:
        raise ValueError("Falló el embebido de la molécula.")
    
    # Guardar estructura 3D sin optimizar
    with Chem.PDBWriter(f"{prefijo}_sin_optimizar.pdb") as w:
        w.write(mol)
    
    # Optimizar geometría
    mmff_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    if mmff_status != 0:
        print("Advertencia: Optimización con MMFF no convergió completamente. Intentando con UFF.")
        uff_status = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        if uff_status != 0:
            print("Advertencia: Optimización con UFF tampoco convergió completamente.")
        else:
            print("Optimización con UFF exitosa.")
    else:
        print("Optimización con MMFF exitosa.")
    
    # Guardar estructura 3D optimizada
    with Chem.PDBWriter(f"{prefijo}_optimizado.pdb") as w:
        w.write(mol)
    
    # Guardar SMILES
    final_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    print("\nSMILES final: ", final_smi)
    with open(f"{prefijo}.smi", "w") as f:
        f.write(final_smi + "\n")

def generar_ciclodextrina(n_units, output_dir):
    output_file = os.path.join(output_dir, f"ciclodextrina_{n_units}.pdb")
    # ... generar y guardar el archivo en output_file ...

if __name__ == "__main__":
    polimero = generar_polimero(6)
    guardar_estructura(polimero)
