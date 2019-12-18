import numpy as np
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors
from IPython.display import SVG
import cairosvg
from  sklearn import preprocessing

def gradient2atom(smi, gradient, pos_cut = 3, neg_cut = -3, nBits = 2048):
    """
    map the gradient of Morgan fingerprint bit on the molecule
    Input:
        smi - the smiles of the molecule (a string)
        gradient - the 2048 coeffients of the feature
        cutoff - if positive, get the pos where the integrated weight is bigger than the cutoff;
                 if negative, get the pos where the integrated weight is smaller than the cutoff
    Output:
        two list of atom ids (positive and negative)   
    """
    # generate mol 
    mol = Chem.MolFromSmiles(smi)
    # get the bit info of the Morgan fingerprint
    bi = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = 2, bitInfo=bi, nBits=nBits)
    onbits = list(fp.GetOnBits())
    # calculate the integrated weight
    atomsToUse = np.zeros((len(mol.GetAtoms()),1))
    for bitId in onbits:
        atomID, radius = bi[bitId][0]
        temp_atomsToUse = []
        if radius > 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
            for b in env:
                temp_atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                temp_atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        else:
            temp_atomsToUse.append(atomID)
            env = None
        temp_atomsToUse = list(set(temp_atomsToUse))
        atomsToUse[temp_atomsToUse] += gradient[bitId]
    # get the postively/negatively contributed atom ids
    highlit_pos = []
    highlit_neg = []
    for i in range(len(atomsToUse)):
        if  atomsToUse[i] > pos_cut:
            highlit_pos.append(i)
        elif atomsToUse[i] < neg_cut:
            highlit_neg.append(i)
    return mol, highlit_pos, highlit_neg, atomsToUse

def color_rendering(atomsToUse, cutoff = 10, clip = 0.5):
    cmap = cm.RdBu_r
    color_dict = {}
    #print(atomsToUse)
    atomsToUse = (atomsToUse.flatten() - clip) / cutoff + 0.5
    for i in range(len(atomsToUse)):
        color_dict[i] = cmap(atomsToUse[i])[0:3]
    return atomsToUse, color_dict

def moltosvg(mol,molSize=(450,200),kekulize=True,drawer=None,**kwargs):
    mc = rdMolDraw2D.PrepareMolForDrawing(mol,kekulize=kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))

#-----------------------------------------

def plot_SAR(compound_df, task_df, chem_idx, task_id, SAR_cnt, 
             mode = 'RobustMT', cutoff = 10, clip = 0.5,
             n_features = 2048):
    smiles = compound_df['canonical_smiles'].iloc[chem_idx]

    grad_flag = True
    if mode == 'RobustMT':
        grad = task_df[task_id].tolist()
    elif mode == 'ST':
        grad = task_df.iloc[chem_idx].tolist()
    elif mode == 'baseline':
        grad = task_df[task_id].tolist()
    elif mode == 'AttentiveFP':
        values =  np.array(task_df.iloc[chem_idx].tolist())
        atomsToUse = np.array([item for item in values if ~(item == 0.0)])
        grad_flag = False

    #grad = preprocessing.scale(np.array(grad).reshape(-1))
    #print(grad.shape)

    if grad_flag:
        mol, highlit_pos, highlit_neg, atomsToUse = gradient2atom(smiles, grad)
    else:
        mol = Chem.MolFromSmiles(smiles)

    cutoff = np.ceil(np.max(np.array([np.absolute(np.min(atomsToUse)), np.max(atomsToUse)])))
    clip = np.mean(atomsToUse)
    #print(cutoff, clip)

    atomsToUse, color_dict = color_rendering(atomsToUse, cutoff, clip)

    img = moltosvg(mol, molSize=(300,200), highlightAtoms=[m for m in range(len(atomsToUse))],
            highlightAtomColors = color_dict, highlightBonds=[])
    cairosvg.svg2png(bytestring=img.data, write_to = 'VISAR_webapp/static/SAR_%d.png' % SAR_cnt)
    return

