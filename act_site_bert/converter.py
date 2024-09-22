import os
from joblib import dump, load

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import decomposition

from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB
from scipy.ndimage import gaussian_filter1d
from ase.io import read,write
from ase.neighborlist import neighbor_list
from ase import Atoms
from ase.build import molecule

from act_site_bert  import PDBTools



def smiles_to_sdf(df, output_directory):
    # SDF 파일을 저장할 디렉토리 확인 및 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for index, row in df.iterrows():
        smiles = row['Ligand SMILES']
        inner_id = row['inner_ID']
        output_sdf_file = os.path.join(output_directory, f"{inner_id}.sdf")
        
        try:
                # SMILES 문자열을 사용하여 분자 객체 생성
            mol = Chem.MolFromSmiles(smiles)
                
            if mol is None:
                    print(f"Invalid SMILES: {smiles}")
                    continue
                
                # 분자에 대한 3D 좌표를 생성
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
                
                # SDF 파일에 분자 객체를 작성
            writer = Chem.SDWriter(output_sdf_file)
            writer.write(mol)
            writer.close()
            
        except Exception as e:
            print(f"{smiles} failed: {str(e)}")
            with open(os.path.join(output_directory, "failed_list.txt"), 'a') as file:
                file.write(f"{smiles}\n")
    
    print(f"Successfully saved molecules to {output_directory}")


def create_docking_sh_script(directory, pdbqt_file, output_sh_file, docking_result_dir, cx, cy, cz):
    # 지정된 디렉토리 내의 모든 파일명을 저장할 리스트
    ligand_files = []
    n = 0
    # 디렉토리 내에서 모든 파일명을 확인
    for filename in os.listdir(directory):
        if filename.endswith(".sdf"):  # 특정 확장자로 필터링, 필요에 따라 수정 가능
            n += 1
            ligand_files.append(filename)
        # if n > 20 :
        #     break
    # sh 파일에 명령어 작성
    with open(output_sh_file, "w") as sh_file:
        sh_file.write("source activate unidock_env")
        for ligand_name in ligand_files:
            ligand_path = os.path.join(directory, ligand_name)
            command = f"unidocktools unidock_pipeline -r '{pdbqt_file}' -l '{ligand_path}' -sd '{docking_result_dir}' -cx {cx} -cy {cy} -cz {cz} \n"
            sh_file.write(command)

    print(f"Shell script created and saved to {output_sh_file}")


def convert_sdf_to_pdb(sdf_file, pdb_file):
    # SDF 파일을 로드합니다.
    try :
        suppl = Chem.SDMolSupplier(sdf_file)
        
        for mol in suppl:
            if mol is None:
                continue
            
            # PDB 파일로 저장합니다.
            with open(pdb_file, 'w') as pdb_out:
                pdb_out.write(Chem.MolToPDBBlock(mol))
                print(f"Successfully converted {sdf_file} to {pdb_file}")
    except :
        print(sdf_file , " failed")


def pdb_to_xyz(pdb_file, xyz_file):
    try :
        # PDB 파서 초기화
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        atoms = []
        
        # PDB 파일에서 모든 원자 정보를 가져옴
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_info = {
                            'element': atom.element,  # 원자 기호
                            'x': atom.coord[0],       # x 좌표
                            'y': atom.coord[1],       # y 좌표
                            'z': atom.coord[2]        # z 좌표
                        }
                        atoms.append(atom_info)
    
        # XYZ 파일로 저장
        with open(xyz_file, 'w') as f:
            f.write(f"{len(atoms)}\n")  # 원자 수
            f.write(f"Converted from {pdb_file}\n")
            for atom in atoms:
                f.write(f"{atom['element']} {atom['x']:.3f} {atom['y']:.3f} {atom['z']:.3f}\n")
    
        print(f"Successfully converted {pdb_file} to {xyz_file}")
    except :
        print("failed : " , pdb_file)



def getRawInputs(types,atoms,x,v):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ellst=open(os.path.join(current_dir, 'elementsem/models/pcakm/ellist.txt'),'r').read().split('\n')
    km = {i: load(f'{current_dir}/elementsem/models/pcakm/'+i+'_kmeans.pkl') for i in ellst}
    pca = {i: load(f'{current_dir }/elementsem/models/pcakm/'+i+'_pca.pkl') for i in ellst}
    i, d = neighbor_list('id', atoms, 10.0, self_interaction=False)
    rdfatoms,ntypes=[],[]
    for k,l in enumerate(atoms):
        el=types[k]
        y = np.zeros(100)
        dist = np.round(d[i==k]*10)
        a,b=np.unique(dist, return_counts=True)
        np.put(y,a.astype(int)-1,b)
        values=gaussian_filter1d(y/v,1)
        num = km[el].predict(pca[el].transform(np.nan_to_num([values],nan=0,posinf=0, neginf=0)))[0]
        ntypes.append(el+str(num))#el2id[el+str(num)]
    return ntypes

def get_residues_center_of_mass(pdb_file, chain_id, residue_numbers):
    # PDB 파서 초기화
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 구조에서 지정한 체인 찾기
    chain = structure[0][chain_id]  # 모델 0, 체인 ID 선택
    
    # 각 잔기의 원자 좌표 추출
    atom_coords = []
    for res_num in residue_numbers:
        residue = chain[res_num]  # 잔기 번호 선택
        for atom in residue:
            atom_coords.append(atom.coord)  # 각 원자의 좌표를 리스트에 추가
    
    # 좌표를 numpy 배열로 변환
    atom_coords = np.array(atom_coords)
    
    # 중심 좌표 계산
    center_of_mass = atom_coords.mean(axis=0)
    
    return center_of_mass

# Target PDB should be in wokring_dir_path
def smiles_to_Token(df , wokring_dir_path, project_name, smiles_sdf_exit = False, docking_exist = True, exisitng_docking_file_path = None ,with_key_residue = False , Target_file_info = None) :
    WORKINGDIR = wokring_dir_path
    PROJECTNAME = project_name

    # smile to sdf file
    output_directory = os.path.join(WORKINGDIR,f"{PROJECTNAME}_sdf_beforedock")
    os.makedirs(output_directory, exist_ok=True)
    # if not smiles_sdf_exit:
    #     smiles_to_sdf(df, output_directory)

    if with_key_residue : 
        if not docking_exist :
            # get binding site coordinate 
            pdb_file = os.path.join(WORKINGDIR, Target_file_info['target_file_name'])
            chain_id = Target_file_info['chain']
            residue_number = Target_file_info['target_bindingsite_num']  
            center_of_mass = get_residues_center_of_mass(pdb_file, chain_id, residue_number)
            print(f"Center of mass for residue {residue_number} in chain {chain_id}: {center_of_mass}")

            # make docking command shell scripts
            directory = os.path.join(WORKINGDIR, f"{PROJECTNAME}_sdf_beforedock")
            pdbqt_file = os.path.join(WORKINGDIR,Target_file_info['target_file_name'].replace('pdb','pdbqt'))
            output_sh_file = os.path.join(WORKINGDIR,"docking_commands.sh")
            docking_result_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_sdf_afterdock")
            os.makedirs(docking_result_dir, exist_ok=True)

            create_docking_sh_script(directory, pdbqt_file, output_sh_file, docking_result_dir, center_of_mass[0], center_of_mass[1], center_of_mass[2])  

            # sdf file to pdb file
            sdf_file_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_sdf_afterdock")
            pdb_file_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_ligand_pdb_result")
            os.makedirs(pdb_file_dir, exist_ok=True)

            for filename in os.listdir(sdf_file_dir):
                sdf_file_path = os.path.join(sdf_file_dir,filename)
                pdb_file_path = os.path.join(pdb_file_dir,filename.replace('sdf','pdb'))
                convert_sdf_to_pdb(sdf_file_path, pdb_file_path)

        
        else :
            exisitng_docking_file_path = exisitng_docking_file_path
            # sdf file to pdb file
            sdf_file_dir = os.path.join(exisitng_docking_file_path)
            pdb_file_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_ligand_pdb_result")
            os.makedirs(pdb_file_dir, exist_ok=True)

            # make pdb with only key residue
            chain_id = Target_file_info['chain']
            residue_numbers =  Target_file_info['key_residue_num']  
            pdb_file = os.path.join(WORKINGDIR, Target_file_info['target_file_name'])
            output_pdb_file = os.path.join(WORKINGDIR, Target_file_info['target_file_name'].split('.')[0] + "_keyresidues.pdb")
            PDBTools.extract_selected_residues(pdb_file, chain_id, residue_numbers, output_pdb_file)            

            for filename in os.listdir(sdf_file_dir):
                sdf_file_path = os.path.join(sdf_file_dir,filename)
                pdb_file_path = os.path.join(pdb_file_dir,filename.replace('sdf','pdb'))
                convert_sdf_to_pdb(sdf_file_path, pdb_file_path)

            # make Ligand + Key Residue pdb file
            ligand_pdb_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_ligand_pdb_result")
            combined_pdb_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_combined_pdb")
            os.makedirs(combined_pdb_dir, exist_ok=True)


            for filename in os.listdir(ligand_pdb_dir):
                ligand_pdb_file_path = os.path.join(ligand_pdb_dir,filename)
                key_residue_file_path = os.path.join(WORKINGDIR, Target_file_info['target_file_name'].split('.')[0] + "_keyresidues.pdb")
                combined_file_path = os.path.join(combined_pdb_dir, filename)
                
                PDBTools.combine_pdb_files(ligand_pdb_file_path, key_residue_file_path,combined_file_path)

        # pdb file to xyz file
        pdf_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_combined_pdb")
        xyz_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_combinded_xyz")
        os.makedirs(xyz_dir, exist_ok=True)

        for filename in os.listdir(pdf_dir):
            pdb_file_path = os.path.join(pdf_dir, filename)
            xyz_file_path = os.path.join(xyz_dir, filename.replace("pdb","xyz"))
            pdb_to_xyz(pdb_file_path, xyz_file_path)

        x= np.arange(0,10,0.1)
        v = np.concatenate([[1],4*np.pi/3*(x[1:]**3 - x[:-1]**3)])

        inner_ID = []
        Tokens = []

        xyz_dir =  os.path.join(WORKINGDIR,f"{PROJECTNAME}_combinded_xyz")
        xyz_dir
        for filename in os.listdir(xyz_dir):
            inner_ID.append(filename.replace('.xyz',''))
            xyz_file_path  = os.path.join(xyz_dir, filename)
            atoms=read(xyz_file_path, format='xyz')
            types=getRawInputs(atoms.get_chemical_symbols(),atoms,x,v)
            Tokens.append(types)

        token_df = pd.DataFrame({
            'inner_ID':inner_ID,
            'combined_token':Tokens
        })
        filterd_token_df = token_df[token_df['combined_token'].apply(len) != 0]
        merge_df = pd.merge(token_df,df, how='left', left_on='inner_ID', right_on='inner_ID')

        return merge_df
    else :

        sdf_file_dir =  '/home/sej7583/ChemAI/IRAK4/ActsiteBert/IRAK4_BindingSite_test/IRAK4_BS_test_sdf_beforedock'
        # sdf_file_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_sdf_beforedock")
        pdb_file_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_ligand_pdb_result")
        os.makedirs(pdb_file_dir, exist_ok=True)

        for filename in os.listdir(sdf_file_dir):
                sdf_file_path = os.path.join(sdf_file_dir,filename)
                pdb_file_path = os.path.join(pdb_file_dir,filename.replace('sdf','pdb'))
                convert_sdf_to_pdb(sdf_file_path, pdb_file_path)

        pdf_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_ligand_pdb_result")
        xyz_dir = os.path.join(WORKINGDIR,f"{PROJECTNAME}_only_ligand_xyz")
        os.makedirs(xyz_dir, exist_ok=True)

        for filename in os.listdir(pdf_dir):

                pdb_file_path = os.path.join(pdf_dir, filename)
                xyz_file_path = os.path.join(xyz_dir, filename.replace("pdb","xyz"))
                pdb_to_xyz(pdb_file_path, xyz_file_path)




        x= np.arange(0,10,0.1)
        v = np.concatenate([[1],4*np.pi/3*(x[1:]**3 - x[:-1]**3)])

        inner_ID = []
        Tokens = []

        xyz_dir =  os.path.join(WORKINGDIR,f"{PROJECTNAME}_only_ligand_xyz")
        for filename in os.listdir(xyz_dir):
            # if filename == 'id_1989.xyz':
            #     continue
    
            inner_ID.append(filename.replace('.xyz',''))
            xyz_file_path  = os.path.join(xyz_dir, filename)
            atoms=read(xyz_file_path, format='xyz')
            types=getRawInputs(atoms.get_chemical_symbols(),atoms,x,v)
            Tokens.append(types)

        token_df = pd.DataFrame({
            'inner_ID':inner_ID,
            'onlyligand_token':Tokens
        })
        filterd_token_df = token_df[token_df['onlyligand_token'].apply(len) != 0]
        merge_df = pd.merge(filterd_token_df,df, how='left', left_on='inner_ID', right_on='inner_ID')

        return merge_df
