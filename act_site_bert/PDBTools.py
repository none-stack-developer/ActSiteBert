from rdkit import Chem
from rdkit.Chem import AllChem

from Bio import PDB

import numpy as np


def extract_selected_residues(pdb_file, chain_id, residue_numbers, output_pdb_file):
    # PDB 파서 초기화
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # 출력 파일을 위한 PDBIO 초기화
    io = PDB.PDBIO()
    
    # 선택한 부분만 필터링하는 함수
    class SelectResidue(PDB.Select):
        def accept_residue(self, residue):
            return residue.id[1] in residue_numbers

        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_model(self, model):
            return True

    # 구조에서 선택한 chain과 residue를 제외한 나머지를 삭제하고 선택된 부분만 저장
    io.set_structure(structure)
    io.save(output_pdb_file, select=SelectResidue())
    print(f"Successfully saved selected residues to {output_pdb_file}")

def combine_pdb_files(pdb_file1, pdb_file2, output_pdb_file):
    try :
        # 첫 번째 PDB 파일 로드
        mol1 = Chem.MolFromPDBFile(pdb_file1, removeHs=False)
        if mol1 is None:
            raise ValueError(f"Failed to load the first PDB file: {pdb_file1}")
    
        # 두 번째 PDB 파일 로드
        mol2 = Chem.MolFromPDBFile(pdb_file2, removeHs=False)
        if mol2 is None:
            raise ValueError(f"Failed to load the second PDB file: {pdb_file2}")
    
        # 두 분자를 병합
        combined_mol = Chem.CombineMols(mol1, mol2)
    
        # 새로운 PDB 파일로 저장
        with open(output_pdb_file, 'w') as pdb_out:
            pdb_out.write(Chem.MolToPDBBlock(combined_mol))
            print(f"Successfully combined {pdb_file1} and {pdb_file2} into {output_pdb_file}")
    except :
        print("failed  : ",  pdb_file1)

def count_atoms_in_pdb(file_path):
    atom_count = 0
    
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_count += 1
    
    return atom_count