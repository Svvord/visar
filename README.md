
# Visar Tutorial

This project aims to train neural networks by compound-protein interactions and provides interpretation of the learned model by interactively showing transformed chemical landscape and visualized SAR for chemicals of interest.

## model training


```python
import os
from Model_training_utils import ST_model_hyperparam_screen, ST_model_training
os.environ['CUDA_VISIBLE_DEVICES']='1'
```


```python
# initialize parameters
task_names = ['T107', 'T108','T51',
     'T106','T105', 'T10618','T227', 'T168', 'T10624', 'T10627', 'T10209']
MT_dat_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'

params_dict = {
    "n_tasks": [1],
    "n_features": [2048], ## need modification given FP types
    "activation": ['relu'],
    "momentum": [.9],
    "batch_size": [128],
    "init": ['glorot_uniform'],
    "learning_rate": [0.01],
    "decay": [1e-6],
    "nb_epoch": [30],
    "dropouts": [.2, .4],
    "nb_layers": [1],
    "batchnorm": [False],
    #"layer_sizes": [(100, 20), (64, 24)],
    "layer_sizes": [(1024, 512),(1024,128) ,(512, 128),(512,64),(128,64),(64,32), 
                    (1024,512,128), (512,128,64), (128,64,32)],
    "penalty": [0.1]
}
```


```python
# initialize model setup
import random
import time
random_seed = random.randint(0,1000)
local_time = time.localtime(time.time())
log_path = './logs/'
RUN_KEY = 'ST_%d_%d_%d_%d' % (local_time.tm_year, local_time.tm_mon, 
                              local_time.tm_mday, random_seed)
os.system('mkdir %s%s' % (log_path, RUN_KEY))
print(RUN_KEY)
```


```python
# hyperparam screening using deepchem
log_output = ST_model_hyperparam_screen(MT_dat_name, task_names, FP_type, params_dict, 
                                        log_path = './logs/'+RUN_KEY)
```


```python
# manually pick the training parameters
best_hyperparams = {'T107': [(512,64,1), 0.4],
                    'T108': [(512,128,1), 0.2],
                    'T10209': [(512,64,1), 0.4],
                    'T105': [(512,128,1), 0.2],
                    'T106': [(512,64,1), 0.2],
                    'T10618': [(512,128,1), 0.4],
                    'T10624': [(512,128,1), 0.2],
                    'T10627': [(512,64,1), 0.2],
                    'T168': [(512,128,1), 0.2],
                    'T227': [(512, 64, 1), 0.4],
                    'T51': [(512, 128, 64,1), 0.2]
                   }
```


```python
# model training
output_df = ST_model_training(MT_dat_name, FP_type, 
                              best_hyperparams, result_path = './logs/'+RUN_KEY)
```

## build landscape and display interactive plot


```python
from Model_landscape_utils import landscape_building
from Model_training_utils import prepare_dataset, extract_clean_dataset
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import pandas as pd
from bokeh.plotting import output_notebook, show
output_notebook()
```


```python
task_name = 'T107'
db_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'
log_path = './logs/MT_2019_4_16_780/'
prev_model = './logs/ST_2019_3_6_697/T107_rep0_50.hdf5'
n_layer = 1
SAR_result_dir = log_path
output_sdf_name = log_path + 'T107_chemical_landscape.sdf'
```


```python
landscape_building(task_name, db_name, log_path, FP_type,
                       prev_model, n_layer, 
                       SAR_result_dir, output_sdf_name)
```


```python
# pick clusters of interest and pack them as an sdf for pharmacophore modeling
from Model_landscape_utils import sdf2df
landscape_sdf_file = './Result/T107_baseline_landscape.sdf'
landscape_df = sdf2df(landscape_sdf_file)

custom_filter = landscape_df['Label'] == 7
df2sdf(df, output_sdf_name, smiles_field, id_field, custom_filter = None)
```


```python
# pharmacophore building
home_dir = './Result/'
os.chdir(home_dir)

# prepare ligand conformations
from rdkit import Chem
from rdkit.Chem import AllChem

raw_sdf_file = 'Label_7.sdf'
sdf_file = home_dir + 'Label7_rdkit_conf.sdf'
ms = [x for x in Chem.SDMolSupplier(raw_sdf_file)]
n_conf = 5
w = Chem.SDWriter(sdf_file)
for i in range(n_conf):
    ms_addH = [Chem.AddHs(m) for m in ms]
    for m in ms_addH:
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMoleculeConfs(m)
        w.write(m)

# process pharmacophores
result_dir = home_dir + 'Label7_rdkit_phars/'
output_name = 'Cluster7_'
proceed_pharmacophore(home_dir, sdf_file, result_dir, output_name)
```


```python
# visualize the pharmacophore model in pymol
```

## analysis of custom chemicals


```python
from Model_landscape_utils import landscape_positioning
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import pandas as pd
from bokeh.plotting import output_notebook
output_notebook()
```


```python
# set custom file
custom_file = './Result/custom_df.csv'
custom_smi_field = "smiles"
custom_id_field = 'molname'
custom_task_field = 'dummy'

# set the landscape to compare to
task_name = 'T107'
db_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'
log_path = './logs/MT_2019_4_16_780/'
prev_model = './logs/ST_2019_3_6_697/T107_rep0_50.hdf5'
n_layer = 1
custom_SAR_result_dir = log_path
custom_sdf_name = log_path + 'custom_chemicals_on_T107_landscape.sdf'
```


```python
landscape_positioning(custom_file, custom_smi_field, custom_id_field, custom_task_field,
                        task_name, db_name, FP_type, log_path,
                        prev_model, n_layer, custom_SAR_result_dir, custom_sdf_name)
```


```python
# pick clusters of interest and pack them as an sdf fur pharmacophore modeling
from Model_landscape_utils import sdf2df
landscape_sdf_file = './Result/T107_baseline_landscape.sdf'
landscape_df = sdf2df(landscape_sdf_file)

custom_filter = landscape_df['Label'] == 7
df2sdf(df, output_sdf_name, smiles_field, id_field, custom_filter = None)
```
