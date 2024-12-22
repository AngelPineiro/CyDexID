# paste_GPUs.py
import numpy as np
from rdkit import Chem
from pathlib import Path
import subprocess
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from math import asin, pi
from scipy.optimize import minimize

def find_radius(lengths, tol=1e-12, max_iter=1000):
    """
    Dado un conjunto de longitudes, encuentra el radio R tal que:
    sum arcsin(l_i/(2R)) = π.
    Se usa un método de búsqueda de raíces (bisección).

    Debe cumplirse que 2R >= max(l_i), de lo contrario arcsin no está definido.
    """
    Lmax = np.max(lengths)
    # El radio mínimo Rmin es tal que l_max = 2Rmin sin(...) pero sin(...) <=1, 
    # así que Rmin = Lmax/2.
    Rmin = Lmax/2.0
    # Un R muy grande hará l_i/(2R) ~ 0, entonces arcsin(...) ~ l_i/(2R),
    # y sum arcsin(...) sera pequeño. Necesitamos que sum arcsin(...) = π, 
    # así que incrementamos hasta un valor grande.
    Rmax = Rmin*1e6  # límite superior grande
    
    def f(R):
        val = 0.0
        for l in lengths:
            ratio = l/(2*R)
            if ratio > 1.0:
                # No factible
                return 1e9
            val += asin(ratio)
        return val - pi
    
    # Búsqueda por bisección
    fmin = f(Rmin)
    fmax = f(Rmax)
    if fmin*fmax > 0:
        # No se encontró cambio de signo, quizás no factible.
        # Intentar aumentar Rmax aún más.
        # O retornar el que minimice el error.
        # Aquí, retornamos None si no hallamos intersección.
        return None
    
    for _ in range(max_iter):
        Rmid = 0.5*(Rmin+Rmax)
        fmid = f(Rmid)
        if abs(fmid) < tol:
            return Rmid
        if fmin*fmid < 0:
            Rmax = Rmid
            fmax = fmid
        else:
            Rmin = Rmid
            fmin = fmid
    return Rmid

def polygon_on_circle(lengths):
    """
    Dado un conjunto de longitudes y un orden fijo (el orden es el del array),
    calcula el polígono inscrito en la circunferencia que minimiza la varianza
    radial (en realidad la elimina por completo), cumpliendo:

    - El polígono es cerrado.
    - Los vértices están en orden angular creciente.
    - Las longitudes se preservan.

    Retorna las coordenadas x, y de los vértices y el radio R.
    """
    R = find_radius(lengths)
    if R is None:
        # No se pudo encontrar un R exacto. Intentaremos minimizar el error.
        # Como fallback, podemos intentar minimizar la función f(R) con un óptimo local.
        # Para simplicidad, retornamos None.
        return None, None, None
    
    # Con R encontrado:
    deltas = []
    for l in lengths:
        delta = 2*asin(l/(2*R))
        deltas.append(delta)
    deltas = np.array(deltas)

    # Ajustar para que sum(deltas)=2π (debería serlo con el R encontrado)
    # Si hay un ligero error numérico, normalizar:
    factor = (2*np.pi)/np.sum(deltas)
    deltas *= factor

    alphas = np.cumsum(deltas) - deltas[0] # Hacer alpha_1 = 0
    # alphas[0]=0, alphas[i] = sum deltas hasta i-1.

    xs = R*np.cos(alphas)
    ys = R*np.sin(alphas)

    return R, xs, ys

class GPUPaster:
    def __init__(self, n_units, output_dir):
        self.n_units = n_units
        self.output_dir = output_dir
        self.gpus = []
        self.static_dir = Path(output_dir)
        self.target_angle = (self.n_units - 2) * 180 / self.n_units
        self.atoms_per_gpu = []  # Lista para guardar el número de átomos de cada GPU

    def read_pdbs(self):
        """Lee los archivos PDB de las unidades."""
        self.gpus = []
        self.atoms_per_gpu = []  # Reset la lista
        for i in range(1, self.n_units + 1):
            pdb_path = Path(self.output_dir) / f"unidad_{i}.pdb"
            print(f"Leído archivo {pdb_path}")
            
            mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=True)
            if mol is None:
                raise ValueError(f"No se pudo leer el archivo {pdb_path}")
            
            mol = Chem.AddHs(mol, addCoords=True)
            self.gpus.append(mol)
            self.atoms_per_gpu.append(mol.GetNumAtoms())  # Guardar número de átomos

    def find_ring_atoms(self, mol):
        """Encuentra el anillo de 6 miembros que contiene C1."""
        rings = mol.GetRingInfo().AtomRings()
        c1_idx = None
        
        # Primero encontrar C1
        for atom in mol.GetAtoms():
            if atom.GetPDBResidueInfo() and atom.GetPDBResidueInfo().GetName().strip() == 'C1':
                c1_idx = atom.GetIdx()
                break
        
        if c1_idx is None:
            raise ValueError("No se encontró el átomo C1")

        # Buscar el anillo que contiene C1
        for ring in rings:
            if len(ring) == 6 and c1_idx in ring:
                return ring
                
        raise ValueError("No se encontró el anillo de 6 miembros que contiene C1")

    def identify_key_atoms(self, mol, pdb_path):
        """
        Identifica C1, O1, Cx y Ox.
        Retorna: (c1_idx, o1_idx, cx_idx, ox_idx, o2_idx)
        """
        ring_atoms = self.find_ring_atoms(mol)
        c1_idx = None
        o1_idx = None
        o2_idx = None
        
        # Encontrar C1, O1 y O2
        for atom in mol.GetAtoms():
            if atom.GetPDBResidueInfo():
                atom_name = atom.GetPDBResidueInfo().GetName().strip()
                if atom_name == 'C1':
                    c1_idx = atom.GetIdx()
                elif atom_name == 'O1':
                    o1_idx = atom.GetIdx()
                elif atom_name == 'O2':
                    o2_idx = atom.GetIdx()

        if None in (c1_idx, o1_idx, o2_idx):
            raise ValueError("No se encontraron C1, O1 u O2")

        # Encontrar Cx (carbono opuesto a C1 en el anillo)
        distances = {}
        for ring_atom_idx in ring_atoms:
            if ring_atom_idx != c1_idx:
                path = Chem.GetShortestPath(mol, c1_idx, ring_atom_idx)
                if path:
                    distances[ring_atom_idx] = len(path)
        
        # El Cx será el átomo del anillo más lejano a C1
        cx_idx = max(distances.items(), key=lambda x: x[1])[0]

        # Encontrar Ox (oxígeno unido a Cx fuera del anillo)
        cx_atom = mol.GetAtomWithIdx(cx_idx)
        ox_idx = None
        for bond in cx_atom.GetBonds():
            other_atom = bond.GetOtherAtom(cx_atom)
            if other_atom.GetSymbol() == 'O' and not other_atom.IsInRing():
                ox_idx = other_atom.GetIdx()
                break

        if ox_idx is None:
            raise ValueError("No se encontró el oxígeno unido a Cx")

        return c1_idx, o1_idx, cx_idx, ox_idx, o2_idx

    def apply_transformation(self, mol, matrix):
        """Aplica una matriz de transformación a una molécula."""
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            pos_array = np.array([pos.x, pos.y, pos.z, 1.0])
            new_pos = matrix.dot(pos_array)
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                new_pos[0], new_pos[1], new_pos[2]))

    def combine_structures(self, transformed_gpus):
        """Combina todas las GPUs en una única estructura manteniendo los hidrógenos."""
        # Inicializar con la primera GPU
        combined_mol = transformed_gpus[0]
        
        # Crear un diccionario para rastrear los átomos por residuo
        residue_atoms = {1: []}
        
        # Asignar residuo 1 a todos los átomos de la primera GPU y guardar sus índices
        for atom in combined_mol.GetAtoms():
            if not atom.GetPDBResidueInfo():
                # Crear nueva información PDB para el átomo
                info = Chem.AtomPDBResidueInfo()
                info.SetResidueNumber(1)
                info.SetResidueName('UNL')
                info.SetName(atom.GetSymbol() + str(atom.GetIdx() + 1))
                info.SetIsHeteroAtom(True)  # Marcar como HETATM
                atom.SetMonomerInfo(info)
            else:
                info = atom.GetPDBResidueInfo()
                info.SetResidueNumber(1)
                info.SetIsHeteroAtom(True)  # Asegurar que es HETATM
            residue_atoms[1].append(atom.GetIdx())
        
        # Combinar con el resto de GPUs asignando números de residuo incrementales
        offset = combined_mol.GetNumAtoms()
        for i, mol in enumerate(transformed_gpus[1:], start=2):
            residue_atoms[i] = []
            
            for atom in mol.GetAtoms():
                if not atom.GetPDBResidueInfo():
                    info = Chem.AtomPDBResidueInfo()
                    info.SetResidueNumber(i)
                    info.SetResidueName('UNL')
                    info.SetName(atom.GetSymbol() + str(atom.GetIdx() + 1))
                    info.SetIsHeteroAtom(True)  # Marcar como HETATM
                    atom.SetMonomerInfo(info)
                else:
                    info = atom.GetPDBResidueInfo()
                    info.SetResidueNumber(i)
                    info.SetIsHeteroAtom(True)  # Asegurar que es HETATM
                residue_atoms[i].append(atom.GetIdx() + offset)
            
            offset += mol.GetNumAtoms()
            combined_mol = Chem.CombineMols(combined_mol, mol)
        
        # Verificar y corregir la asignación de residuos para los hidrógenos
        for atom in combined_mol.GetAtoms():
            if atom.GetAtomicNum() == 1:  # Si es hidrógeno
                parent_residue = None
                parent_atom = None
                for bond in atom.GetBonds():
                    parent_atom = bond.GetOtherAtom(atom)
                    if parent_atom.GetPDBResidueInfo():
                        parent_residue = parent_atom.GetPDBResidueInfo().GetResidueNumber()
                        break
                
                if parent_residue:
                    if not atom.GetPDBResidueInfo():
                        info = Chem.AtomPDBResidueInfo()
                        info.SetResidueNumber(parent_residue)
                        info.SetResidueName('UNL')
                        info.SetName('H' + str(atom.GetIdx() + 1))
                        info.SetIsHeteroAtom(True)  # Marcar como HETATM
                        atom.SetMonomerInfo(info)
                    else:
                        info = atom.GetPDBResidueInfo()
                        current_residue = info.GetResidueNumber()
                        if current_residue != parent_residue:
                            print(f"Corrigiendo H{atom.GetIdx()} de residuo {current_residue} a {parent_residue}")
                            print(f"  Conectado a {parent_atom.GetSymbol()}{parent_atom.GetIdx()}")
                            info.SetResidueNumber(parent_residue)
                            info.SetIsHeteroAtom(True)  # Asegurar que es HETATM
        
        # Verificación final
        for atom in combined_mol.GetAtoms():
            if atom.GetAtomicNum() == 1:  # Si es hidrógeno
                if not atom.GetPDBResidueInfo():
                    print(f"ADVERTENCIA: H{atom.GetIdx()} no tiene información PDB!")
                    continue
                h_residue = atom.GetPDBResidueInfo().GetResidueNumber()
                parent_atom = next(bond.GetOtherAtom(atom) for bond in atom.GetBonds())
                if not parent_atom.GetPDBResidueInfo():
                    print(f"ADVERTENCIA: Átomo padre de H{atom.GetIdx()} no tiene información PDB!")
                    continue
                parent_residue = parent_atom.GetPDBResidueInfo().GetResidueNumber()
                if h_residue != parent_residue:
                    print(f"ERROR: H{atom.GetIdx()} todavía tiene residuo incorrecto!")
                    print(f"  Residuo H: {h_residue}, Residuo padre: {parent_residue}")
                    print(f"  Conectado a {parent_atom.GetSymbol()}{parent_atom.GetIdx()}")
        
        # Guardar el PDB para verificación
        Chem.PDBWriter(f"{self.output_dir}/combined_GPUs.pdb").write(combined_mol)
        return combined_mol

    def form_CD_bonds(self, transformed_gpus):
        """Combina todas las GPUs formando enlaces entre el Ox de cada unidad y el C1 de la siguiente."""
        # Crear una molécula editable
        edit_mol = Chem.RWMol(transformed_gpus[0])
        
        # Mantener un registro de los índices de los átomos clave
        atom_indices = []
        offset = 0  # Para rastrear el desplazamiento de índices
        
        # Primero obtener todos los índices antes de combinar
        for i in range(len(transformed_gpus)):
            mol = transformed_gpus[i]
            c1_idx, _, _, ox_idx, _ = self.identify_key_atoms(mol, f"unidad_{i+1}.pdb")
            atom_indices.append({
                'c1': c1_idx + offset,
                'ox': ox_idx + offset
            })
            offset += mol.GetNumAtoms()
        
        # Ahora combinar las moléculas y formar los enlaces
        for i in range(1, len(transformed_gpus)):
            # Combinar con la siguiente GPU
            edit_mol = Chem.RWMol(Chem.CombineMols(edit_mol.GetMol(), transformed_gpus[i]))
        
        # Formar los enlaces usando los índices guardados
        for i in range(len(transformed_gpus)):
            next_idx = (i + 1) % len(transformed_gpus)
            # Formar enlace entre Ox de la unidad actual y C1 de la siguiente
            edit_mol.AddBond(atom_indices[i]['ox'], 
                            atom_indices[next_idx]['c1'], 
                            Chem.BondType.SINGLE)
        
        # Convertir a molécula final y añadir hidrógenos
        final_mol = edit_mol.GetMol()
        #final_mol = Chem.AddHs(final_mol)
        
        return final_mol

    def save_final_structure(self, mol, filename="combined_GPUs.pdb"):
        """Guarda la estructura final en formato PDB."""
        output_path = os.path.join(self.output_dir, os.path.basename(filename))
        writer = Chem.PDBWriter(output_path)
        writer.write(mol)
        writer.close()

    def plot_polygon(self, xs, ys, R):
        n = len(xs)
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title("Polígono Ajustado a la Circunferencia")
        
        cmap = plt.cm.get_cmap('rainbow', n)
        for i in range(n):
            xseg = [xs[i], xs[(i+1)%n]]
            yseg = [ys[i], ys[(i+1)%n]]
            ax.plot(xseg, yseg, color=cmap(i), linewidth=2)
            ax.plot(xs[i], ys[i], 'o', color=cmap(i))

        # Dibuja la circunferencia
        theta = np.linspace(0,2*np.pi,200)
        xc = R*np.cos(theta)
        yc = R*np.sin(theta)
        ax.plot(xc, yc, 'k--', label='Circunferencia ajustada')

        # Distancias radiales (deben ser ~0 si se resolvió bien)
        for i in range(n):
            r_i = np.sqrt(xs[i]**2 + ys[i]**2)
            dx = xs[i]/r_i if r_i>1e-12 else 0
            dy = ys[i]/r_i if r_i>1e-12 else 0
            x_circle = dx*R
            y_circle = dy*R
            ax.plot([xs[i], x_circle], [ys[i], y_circle], 'r:', linewidth=1)
            dist = abs(r_i-R)
            ax.text((xs[i]+x_circle)/2, (ys[i]+y_circle)/2, f"{dist:.3f}", fontsize=8, color='red')
        
        ax.axhline(0,color='gray',linewidth=1)
        ax.axvline(0,color='gray',linewidth=1)
        ax.legend()
        plt.grid(True)
        plt.show()

    def align_rings(self):
        aligned_gpus = []
        num_gpus = len(self.gpus)
        
        print("\nDistancias O1-Ox:")
        GPU_O1_Ox_distances = []
        for i in range(num_gpus):
            mol = self.gpus[i]
            _, o1_idx, _, ox_idx, _ = self.identify_key_atoms(mol, f"unidad_{i+1}.pdb")
            conf = mol.GetConformer()
            
            o1_pos = np.array([
                conf.GetAtomPosition(o1_idx).x,
                conf.GetAtomPosition(o1_idx).y,
                conf.GetAtomPosition(o1_idx).z
            ])
            ox_pos = np.array([
                conf.GetAtomPosition(ox_idx).x,
                conf.GetAtomPosition(ox_idx).y,
                conf.GetAtomPosition(ox_idx).z
            ])
            
            dist = np.linalg.norm(o1_pos - ox_pos)
            GPU_O1_Ox_distances.append(dist)
            print(f"  GPU {i+1}: {dist:.6f}")

        # Paso 2: Calcular posiciones objetivo (xs, ys) y radio R
        R, xs, ys = polygon_on_circle(GPU_O1_Ox_distances)

        #hacer una permutación cíclica de xs,ys
        xs = np.roll(xs, 1)
        ys = np.roll(ys, 1)
        
        # Paso 3: O1_targets y Ox_targets
        O1_targets = np.array([[xs[i], ys[i], 0.0] for i in range(num_gpus)])
        Ox_targets = np.array([O1_targets[(i + 1) % num_gpus] for i in range(num_gpus)])
        
        # Paso 4: Alinear cada GPU
        for i in range(num_gpus):
            mol = self.gpus[i]
            mol_copy = Chem.Mol(mol)
            conf_copy = mol_copy.GetConformer()
            
            _, o1_idx, _, ox_idx, _ = self.identify_key_atoms(mol_copy, f"unidad_{i+1}.pdb")
            
            o1_pos = np.array([
                conf_copy.GetAtomPosition(o1_idx).x,
                conf_copy.GetAtomPosition(o1_idx).y,
                conf_copy.GetAtomPosition(o1_idx).z
            ])
            ox_pos = np.array([
                conf_copy.GetAtomPosition(ox_idx).x,
                conf_copy.GetAtomPosition(ox_idx).y,
                conf_copy.GetAtomPosition(ox_idx).z
            ])
            
            # 1) Trasladar O1 -> O1_target
            t1 = O1_targets[i] - o1_pos  # vector de traslación
            T1 = np.eye(4)
            T1[:3, 3] = t1
            self.apply_transformation(mol_copy, T1)
            
            # Releer posiciones actualizadas
            o1_new = np.array([
                conf_copy.GetAtomPosition(o1_idx).x,
                conf_copy.GetAtomPosition(o1_idx).y,
                conf_copy.GetAtomPosition(o1_idx).z
            ])
            ox_new = np.array([
                conf_copy.GetAtomPosition(ox_idx).x,
                conf_copy.GetAtomPosition(ox_idx).y,
                conf_copy.GetAtomPosition(ox_idx).z
            ])
            
            # 2) Rotar para alinear vector (O1->Ox) con (O1_target->Ox_target)
            v1 = ox_new - o1_new
            v2 = Ox_targets[i] - O1_targets[i]
            
            # Si la longitud de v1 es casi cero, saltar la rotación
            if np.linalg.norm(v1) < 1e-8:
                # Ox y O1 coinciden; no hay nada que rotar
                aligned_gpus.append(mol_copy)
                continue
            
            # Construir la rotación 3D con la función 'rotation_matrix_from_vectors'
            Rmat = self.rotation_matrix_from_vectors(v1, v2)  # 3x3
            
            # Paso A: trasladar O1 al origen
            T_to_origin = np.eye(4)
            T_to_origin[:3, 3] = -o1_new
            
            # Paso B: rotar
            R4 = np.eye(4)
            R4[:3, :3] = Rmat
            
            # Paso C: trasladar de vuelta
            T_back = np.eye(4)
            T_back[:3, 3] = o1_new
            
            # Aplicar T_to_origin * R4 * T_back
            self.apply_transformation(mol_copy, T_to_origin)
            self.apply_transformation(mol_copy, R4)
            self.apply_transformation(mol_copy, T_back)
            
            aligned_gpus.append(mol_copy)
            
            # Cambiar esta línea:
            # aligned_gpus[i] = self.rotate_ring_to_z_axis(aligned_gpus[i], i)
            # Por esta:
            aligned_gpus[i] = self.rotate_ring_to_z_axis(aligned_gpus[i])
            
            # Guardar PDB para inspección
            writer = Chem.PDBWriter(str(self.static_dir / f"aligned_{i+1}.pdb"))
            writer.write(mol_copy)
            writer.close()
        
        # Paso 5: Guardar estructura combinada
        aligned_mol = self.combine_structures(aligned_gpus)
        #self.save_final_structure(aligned_mol, "static/aligned_GPUs.pdb") # 
        return aligned_gpus
    
    def rotate_ring_to_z_axis(self, mol):
        """
        Rota la molécula para que:
        1. La normal al plano del anillo tenga mínima componente Z
        2. O2 apunte hacia abajo (Z negativo)
        3. El eje O1-Ox se mantenga como eje de rotación
        """
        # Obtener átomos clave
        _, o1_idx, o2_idx, ox_idx, _ = self.identify_key_atoms(mol, "unidad_1.pdb")
        conf = mol.GetConformer()
        
        # Obtener posiciones clave
        o1_pos = np.array([conf.GetAtomPosition(o1_idx).x,
                          conf.GetAtomPosition(o1_idx).y,
                          conf.GetAtomPosition(o1_idx).z])
        o2_pos = np.array([conf.GetAtomPosition(o2_idx).x,
                          conf.GetAtomPosition(o2_idx).y,
                          conf.GetAtomPosition(o2_idx).z])
        ox_pos = np.array([conf.GetAtomPosition(ox_idx).x,
                          conf.GetAtomPosition(ox_idx).y,
                          conf.GetAtomPosition(ox_idx).z])
        
        # Calcular eje de rotación (O1-Ox)
        rotation_axis = ox_pos - o1_pos
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Obtener normal al plano del anillo
        ring_atoms = self.find_ring_atoms(mol)
        ring_points = []
        for idx in ring_atoms:
            pos = conf.GetAtomPosition(idx)
            ring_points.append([pos.x, pos.y, pos.z])
        ring_points = np.array(ring_points)
        
        # Calcular normal al plano del anillo
        ring_center = np.mean(ring_points, axis=0)
        centered_ring = ring_points - ring_center
        _, _, Vh = np.linalg.svd(centered_ring)
        ring_normal = Vh[2]
        ring_normal = ring_normal / np.linalg.norm(ring_normal)
        
        # Buscar el ángulo que minimiza la componente Z de la normal
        best_angle = None
        min_z_component = float('inf')
        
        for angle in np.linspace(0, 2*np.pi, 360):  # Precisión de 1°
            rot_matrix = self.axis_rotation_matrix(rotation_axis, angle)
            
            # Rotar normal del anillo y O2
            rotated_normal = np.dot(rot_matrix, ring_normal)
            rotated_o2 = np.dot(rot_matrix, o2_pos - ox_pos) + ox_pos
            
            # Calcular componente Z de la normal rotada
            z_component = abs(rotated_normal[2])
            
            # Solo considerar orientaciones donde O2 apunta hacia abajo
            if rotated_o2[2] < 0 and z_component < min_z_component:
                min_z_component = z_component
                best_angle = angle
        
        if best_angle is None:
            raise ValueError("No se encontró una orientación válida")
        
        # Aplicar la mejor rotación encontrada
        rot_matrix = self.axis_rotation_matrix(rotation_axis, best_angle)
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            pos_array = np.array([pos.x, pos.y, pos.z])
            new_pos = np.dot(rot_matrix, pos_array - ox_pos) + ox_pos
            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(*new_pos))
        
        print(f"Rotación aplicada: {np.degrees(best_angle):.1f}°")
        print(f"Componente Z de la normal: {min_z_component:.3f}")
        
        return mol
    
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """
        Devuelve la matriz de rotación 3D (3x3) que lleva vec1 -> vec2.
        Ambos vec1 y vec2 son arrays numpy de dimensión 3.
        """
        # Normalizar
        vec1_u = vec1 / np.linalg.norm(vec1)
        vec2_u = vec2 / np.linalg.norm(vec2)
        
        # Eje de rotación = vec1 x vec2
        axis = np.cross(vec1_u, vec2_u)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-8:
            # Si los vectores ya están alineados (o uno es cero),
            # devolvemos la identidad (o la rotación de 180 si son opuestos)
            # Para simplicidad, devolvemos la identidad.
            return np.eye(3)
        axis /= norm_axis
        
        # Ángulo de rotación
        angle = np.arccos(np.dot(vec1_u, vec2_u))
        
        # Fórmula de Rodrigues
        K = np.array([
            [0,         -axis[2],  axis[1]],
            [axis[2],   0,         -axis[0]],
            [-axis[1],  axis[0],   0]
        ])
        I = np.eye(3)
        R = I + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
        return R

    def axis_rotation_matrix(self, axis, theta):
        """
        Crea una matriz de rotación para rotar alrededor de un eje arbitrario.
        """
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        
        return np.array([
            [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
            [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
        ])

    def remove_atoms_for_linking(self, gpus):
        """Elimina los átomos O1 y su hidrógeno, y el hidrógeno de Ox para permitir enlaces entre GPUs."""
        processed_gpus = []
        
        for i, mol in enumerate(gpus):
            # Crear una copia editable de la molécula
            edit_mol = Chem.RWMol(mol)
            
            # Identificar los átomos clave
            _, o1_idx, _, ox_idx, _ = self.identify_key_atoms(mol, f"unidad_{i+1}.pdb")
            
            # Encontrar los hidrógenos conectados a O1 y Ox
            o1_hydrogens = []
            ox_hydrogens = []
            
            for atom in edit_mol.GetAtoms():
                if atom.GetAtomicNum() == 1:  # Si es hidrógeno
                    for bond in atom.GetBonds():
                        other_atom = bond.GetOtherAtom(atom)
                        if other_atom.GetIdx() == o1_idx:
                            o1_hydrogens.append(atom.GetIdx())
                        elif other_atom.GetIdx() == ox_idx:
                            ox_hydrogens.append(atom.GetIdx())
            
            # Eliminar los átomos en orden inverso (de mayor a menor índice)
            # para no afectar los índices de los átomos restantes
            atoms_to_remove = sorted([o1_idx] + o1_hydrogens + ox_hydrogens, reverse=True)
            
            for atom_idx in atoms_to_remove:
                edit_mol.RemoveAtom(atom_idx)
            
            # Convertir de vuelta a molécula normal
            processed_gpus.append(edit_mol.GetMol())
        
        return processed_gpus

    def minimize_overlap(self, gpus):
        """
        Minimiza el solapamiento entre GPUs rotando cada una alrededor de su eje O1-Ox.
        Versión con más fuerza bruta pero manteniendo la eficiencia.
        """
        from scipy.optimize import minimize  # Añadir aquí si prefieres
        
        print("\nIniciando optimización de solapamiento...")
        
        optimized_gpus = []
        for mol in gpus:
            optimized_gpus.append(Chem.Mol(mol))
        
        n_gpus = len(optimized_gpus)
        angle_range = np.radians(30)
        
        # Obtener posiciones y ejes de rotación iniciales
        ox_positions = []
        o1_positions = []
        rotation_axes = []
        atoms_to_check = []  # Pre-calcular átomos relevantes
        
        for i, mol in enumerate(optimized_gpus):
            _, o1_idx, _, ox_idx, _ = self.identify_key_atoms(mol, f"unidad_{i+1}.pdb")
            conf = mol.GetConformer()
            
            o1_pos = np.array([conf.GetAtomPosition(o1_idx).x,
                              conf.GetAtomPosition(o1_idx).y,
                              conf.GetAtomPosition(o1_idx).z])
            ox_pos = np.array([conf.GetAtomPosition(ox_idx).x,
                              conf.GetAtomPosition(ox_idx).y,
                              conf.GetAtomPosition(ox_idx).z])
            
            axis = ox_pos - o1_pos
            axis = axis / np.linalg.norm(axis)
            
            o1_positions.append(o1_pos)
            ox_positions.append(ox_pos)
            rotation_axes.append(axis)
            
            # Pre-calcular átomos relevantes (no excluidos)
            excluded = self.get_excluded_atoms(mol)
            relevant_atoms = []
            for atom_idx in range(mol.GetNumAtoms()):
                if atom_idx not in excluded:
                    pos = conf.GetAtomPosition(atom_idx)
                    relevant_atoms.append({
                        'idx': atom_idx,
                        'pos': np.array([pos.x, pos.y, pos.z]),
                        'symbol': mol.GetAtomWithIdx(atom_idx).GetSymbol()
                    })
            atoms_to_check.append(relevant_atoms)

        # Función de puntuación más agresiva
        def evaluate_configuration(angles):
            """Versión optimizada de la función de evaluación con penalizaciones más fuertes."""
            min_distances = []
            overlap_count = 0
            severe_overlap_count = 0
            
            for i in range(n_gpus):
                # Calcular índices de GPUs adyacentes teniendo en cuenta la naturaleza cíclica
                prev_idx = (i - 1) % n_gpus
                next_idx = (i + 1) % n_gpus
                
                # Rotar GPU actual
                rot_i = self.axis_rotation_matrix(rotation_axes[i], angles[i])
                rotated_atoms_i = [np.dot(rot_i, atom['pos'] - ox_positions[i]) + ox_positions[i] 
                                  for atom in atoms_to_check[i]]
                
                # Comparar con GPU previa
                rot_prev = self.axis_rotation_matrix(rotation_axes[prev_idx], angles[prev_idx])
                rotated_atoms_prev = [np.dot(rot_prev, atom['pos'] - ox_positions[prev_idx]) + ox_positions[prev_idx] 
                                      for atom in atoms_to_check[prev_idx]]
                
                # Comparar con GPU siguiente
                rot_next = self.axis_rotation_matrix(rotation_axes[next_idx], angles[next_idx])
                rotated_atoms_next = [np.dot(rot_next, atom['pos'] - ox_positions[next_idx]) + ox_positions[next_idx] 
                                      for atom in atoms_to_check[next_idx]]
                
                # Calcular distancias con GPU previa
                min_dist_prev = float('inf')
                for pos1 in rotated_atoms_i:
                    distances = np.linalg.norm(np.array(rotated_atoms_prev) - pos1, axis=1)
                    current_min = np.min(distances)
                    min_dist_prev = min(min_dist_prev, current_min)
                    
                    # Contar solapamientos
                    overlap_count += np.sum(distances < 2.5)
                    severe_overlap_count += np.sum(distances < 1.2)
                
                # Calcular distancias con GPU siguiente
                min_dist_next = float('inf')
                for pos1 in rotated_atoms_i:
                    distances = np.linalg.norm(np.array(rotated_atoms_next) - pos1, axis=1)
                    current_min = np.min(distances)
                    min_dist_next = min(min_dist_next, current_min)
                    
                    # Contar solapamientos
                    overlap_count += np.sum(distances < 2.5)
                    severe_overlap_count += np.sum(distances < 1.2)
                
                # Guardar la menor distancia de las dos comparaciones
                min_distances.append(min(min_dist_prev, min_dist_next))
            
            # Puntuación más agresiva
            score = (-np.min(min_distances) * 3.0 +  # Más peso a la distancia mínima
                    overlap_count * 8.0 +            # Penalización fuerte por solapamientos
                    severe_overlap_count * 25.0)     # Penalización muy fuerte por solapamientos severos
            return score

        # Más intentos y más variados
        n_attempts = 5  # Aumentar número de intentos
        best_score = float('inf')
        best_angles = None
        
        # Puntos iniciales más diversos
        initial_points = [
            np.zeros(n_gpus),  # Todo ceros
            np.random.uniform(-angle_range, angle_range, n_gpus),  # Completamente aleatorio
            np.array([angle_range/2 * (-1)**(i) for i in range(n_gpus)]),  # Alternando máximos
            np.array([angle_range * (i % 2) / 2 for i in range(n_gpus)]),  # Patrón escalonado
            np.array([angle_range * np.sin(i*np.pi/n_gpus) for i in range(n_gpus)]),  # Patrón sinusoidal
            np.random.uniform(-angle_range, angle_range, n_gpus),  # Completamente aleatorio
            np.random.uniform(-angle_range, angle_range, n_gpus),  # Completamente aleatorio
            np.random.uniform(-angle_range, angle_range, n_gpus),  # Completamente aleatorio
            np.random.uniform(-angle_range, angle_range, n_gpus),  # Completamente aleatorio
            np.random.uniform(-angle_range, angle_range, n_gpus),  # Completamente aleatorio
        ]

        for attempt, init_angles in enumerate(initial_points):
            print(f"\nIntento {attempt + 1}/{n_attempts}")
            
            # Múltiples optimizaciones desde cada punto inicial
            for sub_attempt in range(3):  # 3 intentos desde cada punto inicial
                result = minimize(
                    evaluate_configuration,
                    init_angles + np.random.normal(0, angle_range/10, n_gpus),  # Pequeña perturbación
                    method='Nelder-Mead',
                    bounds=[(-angle_range, angle_range)] * n_gpus,
                    options={
                        'maxiter': 150,  # Más iteraciones
                        'xatol': 1e-5,   # Mayor precisión
                        'fatol': 1e-5
                    }
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_angles = result.x
                    print(f"Nuevo mejor score: {best_score:.3f}")
                    print(f"Sub-intento: {sub_attempt + 1}/3")
                    angles_str = ", ".join(f"{np.degrees(angle):.1f}" for angle in best_angles)
                    print(f"Ángulos [°]: [{angles_str}]")
                    
                    # Si encontramos una solución muy buena, mostrar más detalles
                    if best_score < 100:  # Umbral arbitrario para "muy buena" solución
                        test_angles = evaluate_configuration(best_angles)
                        print(f"Distancia mínima encontrada: {-test_angles:.3f} Å")

        # Aplicar los mejores ángulos encontrados
        for i, mol in enumerate(optimized_gpus):
            rot_matrix = self.axis_rotation_matrix(rotation_axes[i], best_angles[i])
            for atom_idx in range(mol.GetNumAtoms()):
                pos = mol.GetConformer().GetAtomPosition(atom_idx)
                pos_array = np.array([pos.x, pos.y, pos.z])
                new_pos = np.dot(rot_matrix, pos_array - ox_positions[i]) + ox_positions[i]
                mol.GetConformer().SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(*new_pos))

        return optimized_gpus

    def get_excluded_atoms(self, mol):
        """
        Retorna los índices de todos los átomos que deben excluirse del cálculo de distancias:
        - Átomos del anillo principal
        - Hidrógenos del anillo
        - Oxígenos O1 y Ox
        """
        # Obtener átomos del anillo y sus hidrógenos
        ring_atoms = set(self.find_ring_atoms(mol))
        excluded_atoms = set(ring_atoms)
        
        # Añadir los hidrógenos unidos a los átomos del anillo
        for ring_idx in ring_atoms:
            ring_atom = mol.GetAtomWithIdx(ring_idx)
            for bond in ring_atom.GetBonds():
                other_atom = bond.GetOtherAtom(ring_atom)
                if other_atom.GetAtomicNum() == 1:  # Si es hidrógeno
                    excluded_atoms.add(other_atom.GetIdx())
        
        # Añadir O1
        _, o1_idx, _, _, _ = self.identify_key_atoms(mol, "unidad_1.pdb")
        excluded_atoms.add(o1_idx)
        
        # Añadir Ox
        _, _, _, ox_idx, _ = self.identify_key_atoms(mol, "unidad_1.pdb")
        excluded_atoms.add(ox_idx)
        
        return excluded_atoms

    def run(self):
        """Ejecuta todo el proceso."""
        print("Leyendo archivos PDB...")
        self.read_pdbs()
        
        print("Alineando anillos...")
        self.gpus = self.align_rings()
        
        print("Minimizando solapamiento entre GPUs...")
        final_gpus = self.minimize_overlap(self.gpus)
        
        print("Eliminando átomos para permitir enlaces...")
        final_gpus = self.remove_atoms_for_linking(final_gpus)
        
        self.atoms_per_gpu = [mol.GetNumAtoms() for mol in final_gpus]
        
        print("Combinando estructuras...")
        final_structure = self.combine_structures(final_gpus)
        
        print("Guardando estructura sin minimizar...")
        self.save_final_structure(final_structure, "non_minimized.pdb")
        
        print("Proceso completado.")
        return final_structure

def main(n_units):
    """Función principal para ejecutar el GPUPaster."""
    paster = GPUPaster(n_units)
    return paster.run()

if __name__ == "__main__":
    import sys
    n_units = 8 # int(sys.argv[1]) if len(sys.argv) > 1 else 7
    main(n_units)