from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple

def validate_substitution_pattern(pattern: List[Tuple[str, str]]) -> bool:
    """
    Valida que el patrón de sustitución sea correcto.
    
    Reglas:
    - Los dígitos válidos son: '2', '3', '6' (simples), '5' (2+3), '8' (2+6), '9' (3+6), '1' (2+3+6)
    - No puede haber solapamiento de posiciones
    """
    # Convertir los códigos compuestos a sus posiciones individuales
    position_map = {
        '2': {2}, '3': {3}, '6': {6},
        '5': {2, 3}, '8': {2, 6}, '9': {3, 6},
        '1': {2, 3, 6}
    }
    
    # Verificar que los dígitos sean válidos
    if not all(pos in position_map for _, pos in pattern):
        invalid_positions = [pos for _, pos in pattern if pos not in position_map]
        raise ValueError(f"Posiciones inválidas encontradas: {invalid_positions}")
    
    # Verificar que no haya solapamiento de posiciones
    used_positions = set()
    for _, pos in pattern:
        current_positions = position_map[pos]
        if current_positions & used_positions:
            raise ValueError(f"Solapamiento detectado en posiciones: {current_positions & used_positions}")
        used_positions.update(current_positions)
    
    return True

def decode_substitution_pattern(pattern: List[Tuple[str, str]]) -> dict:
    """
    Convierte el patrón de sustitución al formato requerido por mol2pdb.
    """
    # Mapeo de códigos compuestos a posiciones individuales
    position_map = {
        '2': [(2,)], '3': [(3,)], '6': [(6,)],
        '5': [(2,), (3,)], '8': [(2,), (6,)], '9': [(3,), (6,)],
        '1': [(2,), (3,), (6,)]
    }
    
    result = {}
    for subst_type, pos in pattern:
        for positions in position_map[pos]:
            for p in positions:
                result[p] = subst_type
    
    return result

def mol2pdb(stereo_config: dict, substitution_pattern: List[Tuple[str, str]], output_file="mol.pdb"):
    """
    Genera una molécula de glucopiranosa con la estereoquímica y sustituciones especificadas.
    
    Parámetros:
    stereo_config (dict): Configuración de los centros quirales
        Ejemplo: {1:'alpha', 2:'L', 3:'D', 4:'L', 5:'D'}
    
    substitution_pattern (List[Tuple[str, str]]): Lista de pares (tipo_sustitución, posición)
        Ejemplo: [('SBE', '2'), ('native', '3'), ('ME', '6')]
        Posiciones válidas: '2', '3', '6' (simples)
                          '5' (2+3), '8' (2+6), '9' (3+6)
                          '1' (2+3+6)
    """
    # Validar el patrón de sustitución
    validate_substitution_pattern(substitution_pattern)
    
    # Decodificar el patrón de sustitución al formato anterior
    decoded_pattern = decode_substitution_pattern(substitution_pattern)
    
    # Configuración nativa de estereoquímica
    native_stereo = {
        1: {'beta': '@@', 'alpha': '@'},
        2: {'D': '@@', 'L': '@'},
        3: {'D': '@@', 'L': '@'},
        4: {'D': '@@', 'L': '@'},
        5: {'D': '@@', 'L': '@'}
    }
    
    # Diccionario de sustituyentes disponibles
    substitutions = {
        'native': 'O',
        'SBE': 'OCCCS(=O)(=O)O',
        'ME': 'C',
        'HP': 'O[CH2]CC=O'
    }
    
    print("stereo_config: ", stereo_config)
    print("decoded_pattern: ", decoded_pattern)
    
    # Obtener la configuración estereoquímica para cada centro
    # Este código genera un diccionario con las marcas de estereoquímica para cada centro quiral
    # Para cada posición del 1 al 5:
    # - Si la posición está en stereo_config, usa esa configuración (L/D o alpha/beta)
    # - Si no está en stereo_config, usa 'D' como valor por defecto, excepto para la posición 1 que usa 'beta'
    # - Luego obtiene el símbolo de estereoquímica (@@ o @) de native_stereo según la configuración
    stereo_marks = {
        pos: native_stereo[pos][stereo_config.get(pos, 'D' if pos != 1 else 'beta')]
        for pos in range(1, 6)
    }
    print("stereo_marks: ", stereo_marks)
    # Obtener sustituyentes para cada posición
    sub_2 = substitutions[decoded_pattern.get(2, 'native')]
    sub_3 = substitutions[decoded_pattern.get(3, 'native')]
    sub_6 = substitutions[decoded_pattern.get(6, 'native')]
    
#     # SMILES parametrizado de la glucopiranosa
#     smile = f"[C{stereo_marks[1]}H:1](O)1\
# [C{stereo_marks[2]}H:2]({sub_2})\
# [C{stereo_marks[3]}H:3]({sub_3})\
# [C{stereo_marks[4]}H:4]([O:7])\
# [C{stereo_marks[5]}H:5]([C:6]{sub_6})\
# O1"

    smile = f"[C{stereo_marks[1]}H:1]([O:8])1\
O\
[C{stereo_marks[5]}H:5]([C:6]{sub_6})\
[C{stereo_marks[4]}H:4]([O:7])\
[C{stereo_marks[3]}H:3]({sub_3})\
[C{stereo_marks[2]}H:2]1{sub_2}"
    
    # Generar molécula
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Error al generar molécula desde SMILES: {smile}")
    
    print(f"SMILES generado: {Chem.MolToSmiles(mol)}")
    #Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    
    # Generar estructura 3D
    mol_H = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_H, AllChem.ETKDG()) 
    AllChem.UFFOptimizeMolecule(mol_H)
    
    #optimizar con otro forcefield
    AllChem.MMFFOptimizeMolecule(mol_H)
    
    # Guardar en PDB
    w = Chem.PDBWriter(output_file)
    w.write(mol_H)
    w.close()
    
    # Imprimir información sobre centros quirales
    for at in mol.GetAtoms():
        if at.HasProp('_CIPCode'):
            print(f"Centro quiral en átomo {at.GetIdx()}: {at.GetProp('_CIPCode')}")
    
    return mol

# Ejemplo de uso:
if __name__ == "__main__":
    # Configuración de estereoquímica
    stereo_config = {
        1: 'beta',
        2: 'L',
        3: 'L',
        4: 'D',
        5: 'L'
    }
    
    # Ejemplos de patrones de sustitución válidos:
    patterns = [
        # Sustituciones simples
        [('SBE', '2'), ('ME', '3'), ('native', '6')],
        
        # Sustitución doble en 2 y 3 (usando '5')
        [('SBE', '5'), ('native', '6')],
        
        # Sustitución triple (usando '1')
        [('SBE', '1')],
        
        # Sustitución en 2 y 6 (usando '8')
        [('ME', '8'), ('native', '3')],
        
        # nativa
        [('native', '2'), ('native', '3'), ('native', '6')],
    ]
    
    # Probar cada patrón
    for i, pattern in enumerate(patterns):
        print(f"\nProbando patrón {i + 1}: {pattern}")
        mol = mol2pdb(stereo_config, pattern, f"GPU_{i+1}.pdb")