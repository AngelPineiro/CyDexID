from gen_GPUs import mol2pdb
from rdkit import Chem
from paste_GPUs_06 import GPUPaster

def interpretar_string_canonico(string_canonico):
    """
    Interpreta el string canónico y retorna la información para cada unidad de GPU.
    
    Args:
        string_canonico: str como "SBE_4_7x0066602_HP_3_7x0600080_RS14-0000500_RS236-2600063"
                        Las partes RS14 y RS236 son opcionales, así como las sustituciones
    
    Returns:
        list: Lista de configuraciones para cada unidad
    """
    native_stereo_config  = {1: 'beta',2: 'L', 3: 'L', 4: 'D', 5: 'L'}
    mutated_stereo_config = {1: 'alpha', 2: 'D', 3: 'D', 4: 'L', 5: 'D'}
    
    # Si no hay modificaciones, retornar configuración por defecto
    if not string_canonico or string_canonico.startswith("No"):
        return [{'sustituciones': [], 'stereo_config': {}} for _ in range(7)]  # Por defecto beta-CD (7 unidades)

    partes = string_canonico.split('_')
    
    # Obtener el número de unidades del string canónico
    n_unidades = None
    for parte in partes:
        if 'x' in parte:  # Buscar el patrón Nx donde N es el número de unidades
            n_unidades = len(parte.split('x')[1])
            break
        elif parte.startswith('RS') and '-' in parte:
            # Si no hay x pero hay RS y guión medio, usar longitud de la segunda parte
            siguiente_parte = partes[partes.index(parte) + 1]
            n_unidades = len(siguiente_parte.split('-')[1])
            break
        
    print("substrings into string_canonico= ", partes)
    print("n_unidades: ", n_unidades)
    
    if n_unidades is None:
        raise ValueError("No se pudo determinar el número de unidades del string canónico")
    
    unidades = [{'sustituciones': [], 'stereo_config': {}} for _ in range(n_unidades)]
    
    i = 0
    while i < len(partes):
        # Si no quedan más partes para procesar, salir del bucle
        if i >= len(partes):
            break
        
        if partes[i].startswith('RS'):
            # Procesar modificaciones quirales
            tipo_rs = partes[i].split('-')[0]
            patron = partes[i].split('-')[1]
            
            for pos, digito in enumerate(patron):
                if tipo_rs == 'RS14':
                    if digito in ['1']:
                        unidades[pos]['stereo_config'][1] = mutated_stereo_config[1]
                        unidades[pos]['stereo_config'][4] = native_stereo_config[4]
                    if digito in ['4']:
                        unidades[pos]['stereo_config'][4] = mutated_stereo_config[4]
                        unidades[pos]['stereo_config'][1] = native_stereo_config[1]
                    if digito in ['5']:
                        unidades[pos]['stereo_config'][1] = mutated_stereo_config[1]
                        unidades[pos]['stereo_config'][4] = mutated_stereo_config[4]
                    if digito in ['0']:
                        unidades[pos]['stereo_config'][1] = native_stereo_config[1]
                        unidades[pos]['stereo_config'][4] = native_stereo_config[4]
                else:  # RS236
                    if digito in ['0']:
                        unidades[pos]['stereo_config'][2] = native_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = native_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = native_stereo_config[5]
                    if digito in ['1']:
                        unidades[pos]['stereo_config'][2] = mutated_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = mutated_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = mutated_stereo_config[5]
                    if digito in ['2']:
                        unidades[pos]['stereo_config'][2] = mutated_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = native_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = native_stereo_config[5]
                    if digito in ['3']:
                        unidades[pos]['stereo_config'][2] = native_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = mutated_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = native_stereo_config[5]
                    if digito in ['6']:
                        unidades[pos]['stereo_config'][2] = native_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = native_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = mutated_stereo_config[5]
                    if digito in ['8']:
                        unidades[pos]['stereo_config'][2] = mutated_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = native_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = mutated_stereo_config[5]
                    if digito in ['9']:
                        unidades[pos]['stereo_config'][2] = native_stereo_config[2]
                        unidades[pos]['stereo_config'][3] = mutated_stereo_config[3]
                        unidades[pos]['stereo_config'][5] = mutated_stereo_config[5]
            i += 1
        else:
            # Procesar sustituciones químicas
            # Verificar que existen las siguientes partes antes de acceder
            if i + 2 >= len(partes):
                break
            tipo_sust = partes[i]
            patron = partes[i+2].split('x')[1]
            
            for pos, valor in enumerate(patron):
                if valor != '0':
                    unidades[pos]['sustituciones'].append(
                        (tipo_sust, str(valor))
                    )
            i += 3
            
    for unidad in unidades:
        if not unidad['stereo_config']:  # Si stereo_config está vacío
            unidad['stereo_config'] = native_stereo_config.copy()  # Usar .copy() para evitar referencias compartidas
    
    print("unidades desde interpretar_string_canonico: ", unidades)
    
    return unidades

def main(string_canonico=None):
    """
    Versión modificada de main() que acepta un string_canonico como parámetro
    """
    #antes de empezar, borrar los archivos con extensión .pdb de static
    for file in os.listdir('static'):
        if file.endswith('.pdb'):
            os.remove(os.path.join('static', file))
    
    if string_canonico is None:
        string_canonico = "SBE_4_7x0066602_HP_3_7x0600080_RS14-0000500_RS236-2600063"
    
    try:
        # Interpretar el string canónico
        unidades = interpretar_string_canonico(string_canonico)
        
        print("unidades: ", unidades)
        # Generar cada unidad modificada
        smiles_unidades = []
        # Iteramos sobre cada unidad del polímero
        for i, unidad in enumerate(unidades):
            # Definimos la configuración estereoquímica base
            stereo_config = {
                1: 'beta',  # C1 - Configuración beta
                2: 'L',      # C2 - Configuración L
                3: 'L',      # C3 - Configuración L  
                4: 'D',      # C4 - Configuración D
                5: 'L'       # C5 - Configuración L
            }
            # Actualizamos la configuración base con las especificaciones particulares de esta unidad
            stereo_config.update(unidad['stereo_config']) # Actualiza el diccionario con las configuraciones de estereoquímica
            
            print("stereo_config: ", stereo_config)
            print("sustituciones: ", unidad['sustituciones'])
            mol = mol2pdb(
                stereo_config=stereo_config,
                substitution_pattern=unidad['sustituciones'],
                output_file=f"static/unidad_{i+1}.pdb"
            )
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles_unidades.append(smiles)
        
        # Generar la ciclodextrina usando las unidades modificadas
        #cd_modificada = generar_CD_modificada(smiles_unidades, unidades)
        cd_modificada = GPUPaster(n_units=len(unidades)).position_gpus()
        
        return cd_modificada
        
    except Exception as e:
        raise e

if __name__ == "__main__":
    main() 