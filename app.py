from flask import Flask, request, jsonify
from flask_cors import CORS
from main import interpretar_string_canonico
from gen_GPUs import mol2pdb
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import base64, io, os, time, sys, uuid, subprocess, tempfile
from paste_GPUs_06 import GPUPaster
app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

# Asegurarse de que la carpeta static existe
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/generar_estructura', methods=['POST'])
def generar_estructura():
    try:
        # Generar un ID único para la sesión del usuario
        unique_id = str(uuid.uuid4())
        user_static_path = os.path.join('static', unique_id)
        
        # Crear la carpeta temporal única para el usuario
        if not os.path.exists(user_static_path):
            os.makedirs(user_static_path)
        
        data = request.json
        string_canonico = data.get('string_canonico')
        
        # Interpretar el string canónico
        unidades = interpretar_string_canonico(string_canonico)
        
        print("Iniciando generación de unidades...")
        # Generar cada unidad modificada
        smiles_unidades = []
        for i, unidad in enumerate(unidades):
            print(f"Generando unidad {i+1} de {len(unidades)}...")
            stereo_config = {
                1: 'beta',
                2: 'L',
                3: 'L',
                4: 'D',
                5: 'L'
            }
            stereo_config.update(unidad['stereo_config'])
            
            mol = mol2pdb(
                stereo_config=stereo_config,
                substitution_pattern=unidad['sustituciones'],
                output_file=os.path.join(user_static_path, f"unidad_{i+1}.pdb")
            )
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles_unidades.append(smiles)
        
        print("Iniciando generación de ciclodextrina...", flush=True)
        # Generar la ciclodextrina
        n_units = len(unidades)
        print(f"Número de unidades a generar: {n_units}", flush=True)
        sys.stdout.flush()
        
        paster = GPUPaster(n_units=n_units, output_dir=user_static_path)
        print("GPUPaster inicializado", flush=True)
        sys.stdout.flush()
        
        try:
            print("Iniciando workflow completo de GPUPaster...", flush=True)
            sys.stdout.flush()
            cd_modificada = paster.run()  # El output_dir ya se pasó en el constructor
            print("Workflow de GPUPaster completado", flush=True)
        except Exception as e:
            print(f"Error en GPUPaster: {str(e)}", flush=True)
            raise Exception(f"Error al generar la estructura: {str(e)}")
        
        # Después de generar la estructura
        non_minimized_pdb = os.path.join(user_static_path, 'non_minimized.pdb')
        
        # Esperar a que exista el archivo non_minimized.pdb
        max_wait = 180  # 3 minutos máximo de espera
        start_time = time.time()
        while not os.path.exists(non_minimized_pdb):
            if time.time() - start_time > max_wait:
                raise Exception("Tiempo de espera agotado para la generación de la estructura")
            time.sleep(1)
            print("Esperando generación de archivo PDB...")
        
        print("Archivo PDB generado, procesando estructura...")
        
        # Leer el archivo PDB sin minimizar
        with open(non_minimized_pdb, 'r', encoding='utf-8') as f:
            pdb_data = f.read()
        
        # Verificar si cd_modificada es una molécula válida
        if not isinstance(cd_modificada, Chem.rdchem.Mol):
            cd_modificada = Chem.MolFromPDBFile(non_minimized_pdb)
            if cd_modificada is None:
                raise Exception("No se pudo generar una molécula válida")
        
        # Verificar si la molécula es válida antes de continuar
        if cd_modificada is None:
            raise Exception("No se pudo generar una molécula válida")
            
        # Generar imagen 2D
        try:
            mol_2d = cd_modificada
            AllChem.Compute2DCoords(mol_2d)
        except Exception as e:
            raise Exception(f"Error al generar coordenadas 2D: {str(e)}")
        
        # Mejorar la visualización 2D
        opts = Draw.DrawingOptions()
        opts.bondLineWidth = 3
        opts.atomLabelFontSize = 16
        opts.includeAtomNumbers = False
        opts.additionalAtomLabelPadding = 0.4
        
        # Generar imagen PNG
        img = Draw.MolToImage(mol_2d, size=(1000, 1000), 
            kekulize=True,
            wedgeBonds=True,
            imageType="png",
            fitImage=True,
            options=opts, 
            path=os.path.join(user_static_path, 'estructura.png'))
        
        # Convertir imagen a base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Generar SMILES
        smiles_final = Chem.MolToSmiles(cd_modificada, isomericSmiles=True)
        
        return jsonify({
            'success': True,
            'smiles': smiles_final,
            'imagen_2d': img_str,
            'pdb_noH': pdb_data,
            'pdb_H': pdb_data,
            'mensaje': 'Estructura generada exitosamente',
            'user_dir': user_static_path
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/minimizar_estructura', methods=['POST'])
def minimizar_estructura():
    try:
        data = request.get_json()
        pdb_data = data.get('pdb_data')
        user_dir = data.get('user_dir')
        
        if not pdb_data or not user_dir:
            return jsonify({'success': False, 'error': 'Missing required data'})
        
        if not os.path.exists(user_dir):
            return jsonify({'success': False, 'error': 'Invalid directory'})
        
        # Usar el directorio del usuario para los archivos
        temp_path = os.path.join(user_dir, 'temp_for_min.pdb')
        min_output = os.path.join(user_dir, 'minimized.pdb')
        
        # Crear archivo temporal para la minimización
        with open(temp_path, 'w') as temp_file:
            temp_file.write(pdb_data)
        
        try:
            # Minimizar la estructura manteniendo los hidrógenos existentes
            process = subprocess.run(
                ['obabel', temp_path, '-h', '-O', min_output, '--minimize'],
                timeout=180,
                capture_output=True,
                check=True
            )
            
            # Leer el archivo minimizado
            with open(min_output, 'r') as f:
                minimized_pdb = f.read()
            
            # Limpiar archivo temporal
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'minimized_pdb': minimized_pdb
            })
            
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': 'Minimization timeout'
            })
            
        except subprocess.CalledProcessError as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': f'Minimization error: {str(e)}'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 