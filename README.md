
# VISAR Tutorial

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

    ST_2019_4_20_205
    


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
from Model_landscape_utils import landscape_building, interactive_plot
from Model_training_utils import prepare_dataset, extract_clean_dataset
from keras import backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import pandas as pd
from bokeh.plotting import output_notebook, show
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="bc1a959d-50dd-4424-9069-a276349d4a6e">Loading BokehJS ...</span>
    </div>





```python
task_name = 'T168'
db_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'
log_path = './logs/SAR_2019_4_20/'
prev_model = './logs/2019_4_20_205/T168_rep0_50.hdf5'
n_layer = 1
SAR_result_dir = log_path
output_sdf_name = log_path + 'T168_chemical_landscape.sdf'
```


```python
plot_df = landscape_building(task_name, db_name, log_path, FP_type,
                       prev_model, n_layer, 
                       SAR_result_dir, output_sdf_name)
plot_df.head()
```

    ==== preparing dataset ... ====
    Extracted dataset shape: (396, 3)
    Loading raw samples now.
    shard_size: 8192
    About to start loading CSV from ./logs/MT_2019_4_16_780//temp.csv
    Loading shard 1 of size 8192.
    Featurizing sample 0
    TIMING: featurizing shard 0 took 2.351 s
    TIMING: dataset construction took 2.395 s
    Loading dataset from disk.
    ==== calculating transfer values ... ====
    WARNING:tensorflow:From /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:1108: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    ==== rendering SAR for chemicals on the landscape ... ====
    ==== packing sdf file ... ====
    

    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:410: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if np.issubdtype(type(cell_value), float):
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T168</th>
      <th>molregno</th>
      <th>salt_removed_smi</th>
      <th>coord1</th>
      <th>coord2</th>
      <th>Label</th>
      <th>imgs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2597</th>
      <td>5.990114</td>
      <td>1055209</td>
      <td>CCOC(=O)CCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>19.203522</td>
      <td>8.912773</td>
      <td>7</td>
      <td>./logs/MT_2019_4_16_780/1055209_img.png</td>
    </tr>
    <tr>
      <th>2598</th>
      <td>5.969795</td>
      <td>1055210</td>
      <td>COC(=O)CCCCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>18.263283</td>
      <td>9.447498</td>
      <td>7</td>
      <td>./logs/MT_2019_4_16_780/1055210_img.png</td>
    </tr>
    <tr>
      <th>2599</th>
      <td>5.709955</td>
      <td>1055211</td>
      <td>COC(=O)CCCCCCCCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>18.244911</td>
      <td>9.454099</td>
      <td>7</td>
      <td>./logs/MT_2019_4_16_780/1055211_img.png</td>
    </tr>
    <tr>
      <th>2600</th>
      <td>5.470047</td>
      <td>1055212</td>
      <td>COC(=O)CCCCCCCCCCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>18.221170</td>
      <td>9.463950</td>
      <td>7</td>
      <td>./logs/MT_2019_4_16_780/1055212_img.png</td>
    </tr>
    <tr>
      <th>2601</th>
      <td>5.119977</td>
      <td>1055213</td>
      <td>CC(C)(C)OC(=O)CN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>17.685968</td>
      <td>7.442045</td>
      <td>7</td>
      <td>./logs/MT_2019_4_16_780/1055213_img.png</td>
    </tr>
  </tbody>
</table>




```python
show(interactive_plot(plot_df, x_column = 'coord1', y_column = 'coord2', color_column = task_name, 
                      id_field = 'molregno', value_field = task_name, label_field = 'Label'))
```








  <div class="bk-root" id="126afcf5-d8c2-4bc9-8cfb-3c08970b1ce9"></div>






```python
# pick clusters of interest and pack them as an sdf for pharmacophore modeling
from Model_landscape_utils import df2sdf
output_sdf_name = log_path + 'T168_Cluster12.sdf'
smiles_field = 'salt_removed_smi'
id_field = 'molregno'
custom_df = plot_df.loc[plot_df['Label'] == 12]
df2sdf(custom_df, output_sdf_name, smiles_field, id_field)
```

    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:296: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      frame[molCol] = frame[smilesCol].map(Chem.MolFromSmiles)
    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:410: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if np.issubdtype(type(cell_value), float):
    


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


```python

```

## analysis of custom chemicals


```python
from Model_landscape_utils import landscape_positioning, interactive_plot
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import pandas as pd
from bokeh.plotting import output_notebook, show
output_notebook()
```

    Using TensorFlow backend.
    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="25765e3b-c1e2-46b9-a915-d517997f562b">Loading BokehJS ...</span>
    </div>





```python
# set custom file
custom_file = './data/custom_df.csv'
custom_smi_field = "smiles"
custom_id_field = 'molname'
custom_task_field = 'dummy'

# set the landscape to compare to
landscape_sdf = './logs/SAR_2019_4_20/T168_chemical_landscape.sdf'
task_name = 'T168'
db_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'
log_path = './logs/'
prev_model = './logs/ST_2019_4_20_205/T168_rep0_50.hdf5'
n_layer = 1
custom_SAR_result_dir = log_path
custom_sdf_name = log_path + 'custom_chemicals_on_T168_landscape.sdf'
```


```python
plot_df = landscape_positioning(custom_file, custom_smi_field, custom_id_field, custom_task_field,
                        landscape_sdf, task_name, db_name, FP_type, log_path,
                        prev_model, n_layer, custom_SAR_result_dir, custom_sdf_name)
```

    ==== preparing dataset ... ====
    Extracted dataset shape: (396, 3)
    Loading raw samples now.
    shard_size: 8192
    About to start loading CSV from ./logs/MT_2019_4_16_780//temp.csv
    Loading shard 1 of size 8192.
    Featurizing sample 0
    TIMING: featurizing shard 0 took 2.267 s
    TIMING: dataset construction took 2.303 s
    Loading dataset from disk.
    Extracted dataset shape: (2, 3)
    Loading raw samples now.
    shard_size: 8192
    About to start loading CSV from ./Result/custom_df.csv
    Loading shard 1 of size 8192.
    Featurizing sample 0
    TIMING: featurizing shard 0 took 0.013 s
    TIMING: dataset construction took 0.022 s
    Loading dataset from disk.
    ==== calculating transfer values ... ====
    WARNING:tensorflow:From /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:1108: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    ==== rendering SAR for chemicals on the landscape ... ====
    ==== packing sdf file ... ====
    

    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:410: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if np.issubdtype(type(cell_value), float):
    


```python
plot_df.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coord1</th>
      <th>coord2</th>
      <th>molregno</th>
      <th>T168</th>
      <th>Label</th>
      <th>salt_removed_smi</th>
      <th>imgs</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.690001</td>
      <td>-11.208886</td>
      <td>RIT</td>
      <td>0.000000</td>
      <td>4</td>
      <td>CC1=C(CCN2CCC(=C(c3ccc(F)cc3)c4ccc(F)cc4)CC2)C(=O)N5C=CSC5=N1</td>
      <td>./logs/MT_2019_4_16_780/RIT_img.png</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.472392</td>
      <td>-7.515880</td>
      <td>ERG</td>
      <td>0.000000</td>
      <td>11</td>
      <td>CN1C[C@@H](C=C2[C@H]1Cc3c[nH]c4cccc2c34)C(=O)N[C@]5(C)O[C@@]6(O)[C@@H]7CCCN7C(=O)[C@H](Cc8ccccc8)N6C5=O</td>
      <td>./logs/MT_2019_4_16_780/ERG_img.png</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-12.093712</td>
      <td>14.359370</td>
      <td>1055209</td>
      <td>5.990114</td>
      <td>6</td>
      <td>CCOC(=O)CCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>./logs/MT_2019_4_16_780/1055209_img.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-10.720143</td>
      <td>14.633834</td>
      <td>1055210</td>
      <td>5.969795</td>
      <td>6</td>
      <td>COC(=O)CCCCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>./logs/MT_2019_4_16_780/1055210_img.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-10.718423</td>
      <td>14.634096</td>
      <td>1055211</td>
      <td>5.709955</td>
      <td>6</td>
      <td>COC(=O)CCCCCCCCCN1CCC(CNC(=O)c2c3n(c4ccccc24)CCCO3)CC1</td>
      <td>./logs/MT_2019_4_16_780/1055211_img.png</td>
      <td>0</td>
    </tr>
  </tbody>
</table>




```python
show(interactive_plot(plot_df, 'coord1', 'coord2', task_name, 'molregno', task_name, 'Label'))
```








  <div class="bk-root" id="85057fe6-ad1b-49a5-98fc-8013178423a0"></div>






```python
# pick clusters of interest and pack them as an sdf fur pharmacophore modeling
custom_filter = landscape_df['Label'] == 7
df2sdf(df, output_sdf_name, smiles_field, id_field, custom_filter = None)
```
